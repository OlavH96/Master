from prepare.Downloader import get_observations_with_video
import cv2
import numpy as np


def create_text_to_display(create_text_from, frame_number, text_index):
    total_number_of_frames = number_of_frames
    frames_per = int(total_number_of_frames / len(create_text_from))
    frame_index = int(frame_number / frames_per)
    value = create_text_from[frame_index]

    text_to_write = f'{value}'

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    text_width, text_height = cv2.getTextSize(text_to_write, font, fontScale, lineType)[0]
    topLeft = (0, 10 * text_index + (text_height * (text_index + 1)))
    # Display the resulting frame
    cv2.putText(img=frame,
                text=text_to_write,
                org=topLeft,
                color=fontColor,
                fontFace=font,
                fontScale=fontScale
                )


if __name__ == '__main__':

    observations = get_observations_with_video()

    for o in observations:
        to_observe = [
            o['location_times'],
            o['locations']['coordinates'],
            [a[1] for a in o['address']],
            ['+'+str(int(a[2]))+'m' for a in o['address']],
        ]

        for t in to_observe:
            print(len(t))

        video = o['video_file']
        cap = cv2.VideoCapture(str(video))
        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 1

        while (cap.isOpened()):

            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:

                for i, v in enumerate(to_observe):
                    create_text_to_display(v, frame_number, i)

                cv2.imshow('Frame', frame)
                frame_number += 1
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                exit(1)
        exit(1)
