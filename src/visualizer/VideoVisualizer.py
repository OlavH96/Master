import configparser
import os
from datetime import timedelta
from pathlib import Path

import numpy as np

from src.prepare.Downloader import get_observations_with_video, download_videos_if_not_exists, \
    tie_observations_to_videos, \
    save_observations_as_json, load_observations_from_json
import cv2
from dateutil import parser
from src.sign_detection.image_generation.objectdetection import load_category_index, \
    run_inference_for_single_image, load_frozen_model, run_inference_for_video

from utils import visualization_utils as vis_util
from src.util.Arguments import visualizer_arguments

import math

import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

root_dir = Path.cwd()  # .parent

config = configparser.ConfigParser()
config.read(root_dir / 'config.ini')
output_dir = config['Directories']['downloaded_videos']
output_dir = root_dir / output_dir


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def create_text_to_display(create_text_from, text_index, index):
    value = create_text_from[index]

    text_to_write = f'{value}'

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2
    text_width, text_height = cv2.getTextSize(text_to_write, font, font_scale, line_type)[0]
    top_left = (0, 10 * text_index + (text_height * (text_index + 1)))

    cv2.putText(img=frame,
                text=text_to_write,
                org=top_left,
                color=font_color,
                fontFace=font,
                fontScale=font_scale
                )


def create_data_for_observation(o):
    location_measurements = o['location_measurements']

    parameters_to_include = [
        'DIRECTION',
        # 'SURFACE_STATE_CV',
        # 'ROAD_WEATHER_CV',
        # 'SURFACE_TYPE_CV',
        # 'CRACKING_CV',
        # 'POTHOLING_CV',
        # 'GUARD_RAIL_LEFT_CV',
        # 'GUARD_RAIL_RIGHT_CV',
    ]
    location_measurements = list(
        filter(lambda l: l['parameter_type'] in parameters_to_include, location_measurements))
    location_measurements_values = list(map(lambda measure:
                                            list(
                                                map(lambda rw: f'{measure["parameter_type"]}: {rw}',
                                                    measure['measurements'])
                                            )
                                            ,
                                            location_measurements))

    to_observe = [
        o['location_times'],  # Time
        o['locations']['coordinates'],  # Coordinates
        [a[1] if a != None else '' for a in o['address']],  # Vegreferanse ? område
        [('+' + str(int(a[2])) + 'm') if a != None else '' for a in o['address']],  # vegref + meter
        *location_measurements_values
    ]
    return to_observe


def _get_box(image, detection_box):
    im_width = image.shape[1]
    im_height = image.shape[0]

    (ymin, xmin, ymax, xmax) = detection_box

    (xmin, xmax, ymin, ymax) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)

    return xmin, xmax, ymin, ymax


def _crop_detected_objects_from_image(image, detection_box, data_for_timestep, prediction, score, save_images_dir='./detected_images'):
    box_data = _get_box(image, detection_box)
    (xmin, xmax, ymin, ymax) = box_data

    xmin = math.floor(xmin)
    ymin = math.floor(ymin)
    xmax = math.ceil(xmax)
    ymax = math.ceil(ymax)

    crop_img = image[ymin:ymax, xmin:xmax]
    mean = np.mean(crop_img / 255)
    if mean < 0.1:
        logger.info(f'Discarded image for being too black: score: {score}, mean: {mean}')
        cv2.imwrite(f'./detected_images/ignored_{mean}_.png', crop_img)
        return  # Black image === its a cencored car

    filename = f'{prediction["name"]}_{score}_'
    filename += '_'.join(str(i) for i in data_for_timestep)
    logger.info(f'Detected image {filename}, with mean {mean}')
    cv2.imwrite(f'{save_images_dir}/{filename}_.png', crop_img)
    cv2.imwrite(f'{save_images_dir}/{filename}_fullframe_.png', image)


def analyze_single_frame(frame, num_detections, detection_boxes, detection_classes, detection_scores, categories, data_for_timestep, save_images_dir):
    for i in range(int(num_detections)):
        box = detection_boxes[i]
        prediction = detection_classes[i]
        prediction_score = detection_scores[i]

        if args.extract and prediction_score >= args.extract_limit:
            _crop_detected_objects_from_image(frame, box, data_for_timestep, categories[prediction], prediction_score, save_images_dir=save_images_dir)


def applyCV(data, graph, categories, data_for_timestep):
    if len(data.shape) != 4:
        data = np.expand_dims(data, axis=0)

    output_dict = run_inference_for_single_image(data, graph)

    num_detections = int(output_dict['num_detections'])
    detection_boxes = output_dict['detection_boxes']
    detection_classes = output_dict['detection_classes']
    detection_scores = output_dict['detection_scores']

    data = data[0]
    if num_detections > 0:

        analyze_single_frame(data, num_detections, detection_boxes, detection_classes, detection_scores, categories,
                             data_for_timestep)

        # for i in range(num_detections):
        #     box = detection_boxes[i]
        #     prediction = detection_classes[i]
        #     prediction_score = detection_scores[i]

        #     if args.extract and prediction_score >= args.extract_limit:
        #
        #         _crop_detected_objects_from_image(data, box, data_for_timestep, categories[prediction], prediction_score)

        if args.visual:
            vis_util.visualize_boxes_and_labels_on_image_array(
                data,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                categories,
                # instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)

    return data


def video_to_np(observation):
    return video_path_to_np(str(observation['video_file']))


def video_path_to_np(path):
    cap = cv2.VideoCapture(str(path))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount and ret):
        ret, data = cap.read()
        if ret and type(data) != type(None):
            buf[fc] = data
        fc += 1

    cap.release()
    return buf


def analyze_raw_video_results(video_np, output_dict, categories, frame_index_start):
    num_detections = output_dict['num_detections']
    detection_boxes = output_dict['detection_boxes']
    detection_classes = output_dict['detection_classes']
    detection_scores = output_dict['detection_scores']

    for i, (video_frame, num_detect, detection_box, detection_class, detection_score) in enumerate(
            zip(video_np, num_detections, detection_boxes, detection_classes, detection_scores)):
        data_for_timestep = [f'Frame-{frame_index_start + i}']
        analyze_single_frame(video_frame, num_detect, detection_box, detection_class, detection_score, categories,
                             data_for_timestep,
                             save_images_dir='./detected_images_raw')


def analyze_video_results(video_np, output_dict, categories, observation, to_observe, frame_index_start, frame_total):
    num_detections = output_dict['num_detections']
    detection_boxes = output_dict['detection_boxes']
    detection_classes = output_dict['detection_classes']
    detection_scores = output_dict['detection_scores']
    # logger.info(f'{num_detections}, {len(num_detections)}')
    # logger.info(f'{detection_boxes}, {len(detection_boxes)}')
    # logger.info(f'{detection_classes}, {len(detection_classes)}')
    # logger.info(f'{detection_scores}, {len(detection_scores)}')

    for i, (video_frame, num_detect, detection_box, detection_class, detection_score) in enumerate(
            zip(video_np, num_detections, detection_boxes, detection_classes, detection_scores)):
        # logger.debug(f'Frame {i} / {len(video_np)}, num-detections: {num_detect}')
        data_for_timestep = data_for_frame_from_observation(observation, to_observe, frame_index_start + i, frame_total)
        analyze_single_frame(video_frame, num_detect, detection_box, detection_class, detection_score, categories,
                             data_for_timestep)


def data_for_frame_from_observation(observation, to_observe, frame_index, frame_total):
    location_times = observation['location_times']
    location_times = [parser.parse(t) for t in location_times]
    start_time = location_times[0]

    total_time_seconds = 60 * 5  # Assume constant video length of 5 minutes
    time_per_frame = total_time_seconds / number_of_frames  # Assume constant frame time
    new_time = start_time + timedelta(seconds=time_per_frame * frame_index)
    nearest_time = nearest(location_times, new_time)
    time_index = location_times.index(nearest_time)

    data_for_timestep = [v[time_index] for v in to_observe[1:]]  # Data excluding time
    return data_for_timestep


def analyze_raw_videos(paths):
    graph = load_frozen_model()
    categories = load_category_index()

    for video in paths:

        videodata = video_path_to_np(video)
        split_size = 500  # frames in a batch
        num_chunks = len(videodata) // split_size

        print(len(videodata))
        print(num_chunks)
        split_video = np.array_split(videodata, num_chunks)
        import time

        frame_start_index = 0
        for i, data in enumerate(split_video):
            start = time.time()
            logger.info(f'Starting batch inference {i} / {len(split_video)}, number of frames = {len(data)}')
            o_dict = run_inference_for_video(data, graph)
            logger.info(f'Completed batch inference {i} / {len(split_video)}, time used = {time.time() - start}s')
            logger.info(f'Starting batch analysis {i} / {len(split_video)}')
            analyze_raw_video_results(
                video_np=data,
                output_dict=o_dict,
                categories=categories,
                frame_index_start=frame_start_index
            )
            logger.info(f'Completed batch analysis {i} / {len(split_video)}')
            frame_start_index += len(data)


if __name__ == '__main__':
    args = visualizer_arguments()
    print("Arguments", args)

    if args.raw_videos_path:
        data = os.listdir(args.raw_videos_path)
        data = [os.path.join(args.raw_videos_path, d) for d in data]
        print("Raw videos", data)
        analyze_raw_videos(data)
        exit(0)

    if not args.cached:
        observations = get_observations_with_video(limit=100)
        download_videos_if_not_exists(observations)
        tie_observations_to_videos(observations)
        save_observations_as_json(observations)
    else:
        observations = load_observations_from_json()

    if not args.cached:
        exit(1)

    graph = load_frozen_model()
    categories = load_category_index()

    print('Categories', categories)
    for i, o in enumerate(observations[args.num_vids_to_skip: args.num_vids_to_analyze + args.num_vids_to_skip]):
        logger.info(f"Analyzing video {i + args.num_vids_to_skip}/{len(observations)} {o['_id']}")

        location_times = o['location_times']
        location_times = [parser.parse(t) for t in location_times]
        start_time = location_times[0]
        to_observe = create_data_for_observation(o)

        video = o['video_file']

        cap = cv2.VideoCapture(str(video))
        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_time_seconds = 60 * 5  # Assume constant video length of 5 minutes
        time_per_frame = total_time_seconds / number_of_frames  # Assume constant frame time

        videodata = video_to_np(o)
        split_size = 500  # frames in a batch
        num_chunks = len(videodata) // split_size
        split_video = np.array_split(videodata, num_chunks)
        import time

        frame_start_index = 0
        for i, data in enumerate(split_video):
            start = time.time()
            logger.info(f'Starting batch inference {i} / {len(split_video)}, number of frames = {len(data)}')
            o_dict = run_inference_for_video(data, graph)
            logger.info(f'Completed batch inference {i} / {len(split_video)}, time used = {time.time() - start}s')
            logger.info(f'Starting batch analysis {i} / {len(split_video)}')
            analyze_video_results(data, o_dict, categories, o, to_observe, frame_start_index, number_of_frames)
            logger.info(f'Completed batch analysis {i} / {len(split_video)}')
            frame_start_index += len(data)

        continue

        frameCounter = 0
        new_time = start_time
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                logger.debug(f'Frame {frameCounter} / {number_of_frames}')
                frameCounter += 1

                new_time = new_time + timedelta(seconds=time_per_frame)
                nearest_time = nearest(location_times, new_time)
                time_index = location_times.index(nearest_time)

                data_for_timestep = [v[time_index] for v in to_observe[1:]]  # Data excluding time

                result = applyCV(frame, graph, categories, data_for_timestep)
                if args.visual:
                    for i, v in enumerate(to_observe):
                        create_text_to_display(v, i, time_index)

                    cv2.imshow('Frame', frame)

                    key = cv2.waitKey(25)
                    if key == ord('q'):
                        break
                    if key == ord('e'):
                        exit(1)
                    if key == ord('p'):
                        waiting = True
                        while waiting:
                            if cv2.waitKey(25) & 0xFF == ord('p'):
                                waiting = False
            else:
                break
