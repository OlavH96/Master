import os
import glob
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def load_images(path):
    image_list = []
    image_names = []
    for filename in glob.glob(path):
        im = Image.open(filename)
        image_list.append(im)
        image_names.append(filename)
    return image_list, image_names


def center_image_with_padding(image, x, y):
    old_size = image.size  # old_size[0] is in (width, height) format

    ratio = float(x) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    delta_w = x - new_size[0]
    delta_h = y - new_size[1]
    padding = (delta_w / 2,  # left
               delta_h / 2,  # top
               delta_w - (delta_w / 2),  # right
               delta_h - (delta_h / 2))  # bottom
    padding = tuple(map(int, padding))
    new_im = ImageOps.expand(image, padding)
    x_remain = x - new_im.size[0]
    y_remain = y - new_im.size[1]

    if x_remain:
        new_im = ImageOps.expand(new_im, (x_remain, 0, 0, 0))
    if y_remain:
        new_im = ImageOps.expand(new_im, (0, y_remain, 0, 0))
    return new_im


def load_images_centered():
    path = '../data/signs/*.png'

    images, names = load_images(path)
    filtered_images = []
    filtered_names = []

    for i, n in zip(images, names):
        im = np.array(i)
        if im.shape[-1] != 4:
            continue
        filtered_images.append(i)
        filtered_names.append(n)

    max_x = max([i.size[0] for i in filtered_images])
    max_y = max([i.size[1] for i in filtered_images])

    fitted_images = list(map(lambda i: center_image_with_padding(i, max_x, max_y), filtered_images))
    np_images = np.array([np.array(i) for i in fitted_images])
    return np_images
