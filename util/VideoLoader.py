import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images

def create_images_from_video():
    count = 0
    videoFile = "../data/test.mp4"
    cap = cv2.VideoCapture(videoFile)  # capturing the video from the given path
    frameRate = cap.get(5)  # frame rate
    x = 1
    while (cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read() ## Frame is the image data we need
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename = "../output/frames/frame%d.png" % count;
            count += 1
            cv2.imwrite(filename, frame)
    cap.release()
    print("Done!")

def load_images(path, batch_size=10):
    cap = cv2.VideoCapture(path)  # capturing the video from the given path
    batch = []
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read() ## Frame is the image data we need
        batch.append(frame)
        if (ret != True):
            break
        if i % batch_size == 0:
            yield np.array(batch)
            batch = []
        i+=1
    cap.release()

if __name__ == '__main__':
    # create_images_from_video()
    for i in load_images():
        print(i.shape)
        plt.imshow(i)
        plt.show()