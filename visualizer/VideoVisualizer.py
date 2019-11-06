import configparser
from pathlib import Path
from datetime import timedelta
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np

from prepare.Downloader import get_observations_with_video, download_videos_if_not_exists, tie_observations_to_videos, \
    save_observations_as_json, load_observations_from_json
import cv2
from dateutil import parser
from sign_detection.image_generation.objectdetection import download_model, load_category_index, \
    run_inference_for_single_image, load_frozen_model
from utils import label_map_util

from utils import visualization_utils as vis_util
from matplotlib import pyplot as plt
import skvideo.io
import math

import argparse
import logging as logger

aparser = argparse.ArgumentParser(description='Run video analysis.')
aparser.add_argument('--no-visual', dest='visual', action='store_false', help='Run without displaying video feed')
aparser.set_defaults(visual=True)

aparser.add_argument('--do-not-use-cached', dest='cached', action='store_false',
                     help='Run using downloaded observations and videos')
aparser.set_defaults(cached=True)

aparser.add_argument('--extract-detected', dest='extract', action='store_true',
                     help='Extract and save detected objects')
aparser.set_defaults(extract=False)

aparser.add_argument('--extraction-certainty', dest='extract_limit', type=float,
                     help='Prediction certainty for extracting image')
aparser.set_defaults(extract_limit=0.5)

args = aparser.parse_args()

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
        [a[1] for a in o['address']],  # Vegreferanse ? område
        ['+' + str(int(a[2])) + 'm' for a in o['address']],  # vegref + meter
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


def _crop_detected_objects_from_image(image, detection_box, data_for_timestep, prediction, score):
    box_data = _get_box(image, detection_box)
    (xmin, xmax, ymin, ymax) = box_data
    xmin = math.floor(xmin)
    ymin = math.floor(ymin)
    xmax = math.ceil(xmax)
    ymax = math.ceil(ymax)

    crop_img = image[ymin:ymax, xmin:xmax]
    if np.mean(crop_img) < 0.2: 
        logger.info(f'Discarded image for being too black: score: {score}, mean: {np.mean(crop_img)}')
        cv2.imwrite(f'./detected_images/ignored_{np.mean(crop_img)}_.png', crop_img)
        return  # Black image === its a cencored car

    filename = f'{prediction["name"]}_{score}_'
    filename += '_'.join(str(i) for i in data_for_timestep)
    cv2.imwrite(f'./detected_images/{filename}_.png', crop_img)

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

        for i in range(num_detections):
            box = detection_boxes[i]
            prediction = detection_classes[i]
            prediction_score = detection_scores[i]

            if args.extract and prediction_score >= args.extract_limit:
                
                _crop_detected_objects_from_image(data, box, data_for_timestep, categories[prediction], prediction_score)

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


if __name__ == '__main__':
    print("Arguments", args)

    if not args.cached:
        observations = get_observations_with_video(limit=100)
        download_videos_if_not_exists(observations)
        tie_observations_to_videos(observations)
        save_observations_as_json(observations)
    else:
        observations = load_observations_from_json()

    graph = load_frozen_model()
    categories = load_category_index()

    print('Categories', categories)
    for i,o in enumerate(observations):
        logger.info(f"Analyzing video {i}/{len(observations)}")

        location_times = o['location_times']
        location_times = [parser.parse(t) for t in location_times]
        start_time = location_times[0]
        to_observe = create_data_for_observation(o)

        video = o['video_file']

        cap = cv2.VideoCapture(str(video))
        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_time_seconds = 60 * 5  # Assume constant video length of 5 minutes
        time_per_frame = total_time_seconds / number_of_frames  # Assume constant frame time

        new_time = start_time
        while (cap.isOpened()):

            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
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
