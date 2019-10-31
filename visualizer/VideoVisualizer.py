import configparser
from pathlib import Path
from datetime import timedelta

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

root_dir = Path.cwd()#.parent

config = configparser.ConfigParser()
config.read(root_dir / 'config.ini')
output_dir = config['Directories']['downloaded_videos']
output_dir = root_dir / output_dir


def create_black_box(num_texts):
    cv2.rectangle(frame,
                  (0, 0),
                  (frame.shape[1], (num_texts + 1) * 30),
                  (0, 0, 0),
                  cv2.FILLED)


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
    for l in location_measurements:
        print(l)
    print()
    parameters_to_include = [
        'DIRECTION',
        'SURFACE_STATE_CV',
        'ROAD_WEATHER_CV',
        'SURFACE_TYPE_CV',
        'CRACKING_CV',
        'POTHOLING_CV',
        'GUARD_RAIL_LEFT_CV',
        'GUARD_RAIL_RIGHT_CV',
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


def applyCV(data, graph, categories):

    if len(data.shape) != 4:
        data = np.expand_dims(data, axis=0)

    output_dict = run_inference_for_single_image(data, graph)
    print("Categories:",categories)
    print(output_dict.keys())
    print("num detections", output_dict['num_detections'])
    data = data[0]
    vis_util.visualize_boxes_and_labels_on_image_array(
        data,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        categories,
        #instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    return data


if __name__ == '__main__':

    # observations = get_observations_with_video()
    # download_videos_if_not_exists(observations)
    # tie_observations_to_videos(observations)
    # save_observations_as_json(observations)

    graph = load_frozen_model()
    categories = load_category_index()
    observations = load_observations_from_json()

    for o in observations:

        location_times = o['location_times']
        location_times = [parser.parse(t) for t in location_times]
        start_time = location_times[0]
        to_observe = create_data_for_observation(o)

        video = o['video_file']

        cap = cv2.VideoCapture(str(video))
        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_time_seconds = 60 * 5
        time_per_frame = total_time_seconds / number_of_frames

        new_time = start_time
        while (cap.isOpened()):

            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                new_time = new_time + timedelta(seconds=time_per_frame)
                nearest_time = nearest(location_times, new_time)
                time_index = location_times.index(nearest_time)

                result = applyCV(frame, graph, categories)
                # create_black_box(len(to_observe))

                for i, v in enumerate(to_observe):
                    create_text_to_display(v, i, time_index)

                # plt.imshow(result)
                # plt.savefig('test.png')
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

            # Break the loop
            else:
                exit(1)
