import glob

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt


def load_images(path, num=1000):
    image_list = []
    image_names = []
    for filename in glob.glob(path)[:num]:
        im = Image.open(filename)
        image_list.append(im)
        image_names.append(filename)
    return image_list, image_names


def center_image_with_padding(image, x, y):
    old_size = image.size  # old_size[0] is in (width, height) format
    delta_w = abs(x - old_size[0])
    delta_h = abs(y - old_size[1])
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


def resize_image(image, new_x, new_y):
    return image.resize((new_x, new_y), Image.NEAREST)


def find_max_min_image_size(path='detected_images/*.png'):
    max_x = 0
    max_y = 0

    for i in load_images_generator(path):
        w, h = i.size
        if w > max_x:
            max_x = w
        if h > max_y:
            max_y = h
        del i

    return max_x, max_y


def load_images_centered():
    path = 'detected_images/*.png'

    images, names = load_images(path)

    max_x = max([i.size[0] for i in images])
    max_y = max([i.size[1] for i in images])

    fitted_images = list(map(lambda i: center_image_with_padding(i, max_x, max_y), images))
    return np.array([np.array(i) for i in fitted_images])


def load_images_generator(path='detected_images/*.png', color_mode='RGB'):
    for filename in glob.glob(path):
        im = Image.open(filename).convert(color_mode)
        yield im

def load_images_generator_with_filename(path='detected_images/*.png', color_mode='RGB'):
    for filename in glob.glob(path):
        im = Image.open(filename).convert(color_mode)
        yield im, filename

def load_images_centered_generator(max_x=1280, max_y=720, path='detected_images/*.png'):
    for filename in glob.glob(path):
        im = Image.open(filename)
        centered = center_image_with_padding(im, max_x, max_y)
        yield np.array(centered)


if __name__ == '__main__':

    # import time
    # start = time.time()
    # np_images = load_images_centered()
    # end = time.time()
    # print(end-start, "s")

    for i in load_images_centered_generator():
        plt.imshow(i)
        plt.show()
