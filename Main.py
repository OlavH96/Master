# set the matplotlib backend so figures can be saved in the background
import keras
import matplotlib

from util import VideoLoader

import tensorflow as tf
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
import keras.layers as layers
from keras.layers.pooling import AveragePooling2D
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.transform import resize
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os


def autoencoder(image_shape):
    image_x = image_shape[0]
    image_y = image_shape[1]
    image_z = image_shape[2]

    total_pixels = image_y * image_x * image_z
    translator_factor = 2
    translator_layer_size = int(total_pixels / translator_factor)
    middle_factor = 2
    middle_layer_size = int(translator_layer_size / middle_factor)

    print(total_pixels)
    print(translator_layer_size)
    print(middle_layer_size)

    inputs = keras.Input(shape=image_shape, name='cat_image')
    x = layers.Flatten(name='flattened_cat')(inputs)  # turn image to vector.

    x = layers.Dense(translator_layer_size, activation='relu', name='encoder')(x)
    x = layers.Dense(middle_layer_size, activation='relu', name='middle_layer')(x)
    x = layers.Dense(translator_layer_size, activation='relu', name='decoder')(x)

    outputs = layers.Dense(total_pixels, activation='relu', name='reconstructed_cat')(x)
    outputs = layers.Reshape(image_shape)(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    customAdam = keras.optimizers.Adam(lr=0.001)  # you have no idea how many times I changed this number

    model.compile(optimizer=customAdam,  # Optimizer
                  # Loss function to minimize
                  loss="mean_squared_error",
                  # List of metrics to monitor
                  metrics=["mean_squared_error"])
    return model


if __name__ == '__main__':

    config = tf.ConfigProto(device_count={'CPU': 2})
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    scale_factor = 30
    actual_image_shape = (720, 1280, 3)
    image_shape = (
        int(actual_image_shape[0] / scale_factor),
        int(actual_image_shape[1] / scale_factor),
        actual_image_shape[2]
    )
    epochs = 1
    model = autoencoder(image_shape)
    path = 'data/test.mp4'

    def image_generator():
        for i in VideoLoader.load_images(path=path):
            i = i / 255
            i = resize(i, (1, *image_shape))
            yield (i, i)


    epoch_steps = len(os.listdir('output/frames'))
    model.fit_generator(image_generator(), epochs=epochs, steps_per_epoch=10)

    model.save('model.h5')
    for i in VideoLoader.load_images(path=path):
        plt.imshow(i)
        plt.show()

        i = i / 255
        i = resize(i, (1, *image_shape))
        pred = model.predict(i)
        for p in pred:
            p = resize(p, actual_image_shape)
            plt.imshow(p)
            plt.show()
        break
