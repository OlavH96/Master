# set the matplotlib backend so figures can be saved in the background
import keras
import matplotlib

from util import VideoLoader
import glob
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
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
from keras.models import load_model
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Deconvolution2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.transform import resize
from imutils import paths
import matplotlib.pyplot as plt
from util.ImageLoader import load_images_centered, load_images_centered_generator, load_images_generator, find_max_min_image_size
import numpy as np
import argparse
import pickle
import cv2
import logging as log

# https://medium.com/analytics-vidhya/building-a-convolutional-autoencoder-using-keras-using-conv2dtranspose-ca403c8d144e
def conv_autoencoder(io_shape):
    image_x = io_shape[0]
    image_y = io_shape[1]
    image_z = io_shape[2]
    num_reductions = 3
    
    num_filters_low = 16
    num_filters_high = 32
    

    inp = Input(io_shape)
    e = Conv2D(num_filters_low, (3, 3), activation='relu', padding="same")(inp)
    e = MaxPooling2D((2, 2), strides=2, padding="same")(e)
    
    for _ in range(num_reductions):
        e = Conv2D(num_filters_high, (3, 3), activation='relu', padding="same")(e)
        e = MaxPooling2D((2, 2), strides=2, padding="same")(e)

    e = Conv2D(num_filters_high, (3, 3), activation='relu', padding="same")(e)

    s = [int(i) for i in e.shape[1:]]
    #s[-1] = 1
    prod_s = np.prod(s)
    dense_size = prod_s
    reshape_shape = s

    l = Flatten()(e)
    l = Dense(dense_size, activation='softmax')(l)

    #DECODER
    d = Reshape(reshape_shape)(l)
    
    for _ in range(num_reductions+1):
        d = Conv2DTranspose(num_filters_high,(3, 3), strides=2, activation='relu', padding='same')(d)

    d = Conv2DTranspose(num_filters_low,(3, 3), activation='relu', padding='same')(d)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(d)
    ae = Model(inp, decoded)
    ae.compile(optimizer="adam", loss="mse")
    ae.summary()
    
    assert ae.output_shape[1:] == io_shape, f'Output Shape {decoded.shape} is not equal input shape {io_shape}'
    return ae

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

def train_on_images():
    sess = tf.Session()
    keras.backend.set_session(sess)

    #max_x = max([i.shape[0] for i in images])
    #max_y = max([i.shape[1] for i in images])
    max_x = 512 #304 # 1280 , nearest power of 2
    max_y = 512 # 298 # 720
    #max_x, max_y = find_max_min_image_size(path = 'detected_images/*.png')
    #print(max_x, max_y) # 304, 298

    epochs = 10
    shape = (max_y, max_x, 3)
    model = conv_autoencoder(shape)#autoencoder(shape)
    path = 'detected_images/362.*'
    steps = len(glob.glob(path))
    # define the checkpoint
    filepath = "model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    def centered_image_generator():
        for i in load_images_centered_generator(max_x, max_y, path):
            i = np.expand_dims(i, axis=0)
            i = i / 255
            yield (i,i)
    log.info('Fitting model...')
    model.fit_generator(centered_image_generator(), epochs=epochs, steps_per_epoch=int(steps/epochs), callbacks=callbacks_list)
    log.info('Finished fitting model')
    model.save("autoencoder.h5")

def load_model_and_predict():

    model = load_model('model.h5')
    
    max_x = 512 #304 # 1280 , nearest power of 2
    max_y = 512 # 298 # 720
    path = 'detected_images/362.*'

    def centered_image_generator():
        for i in load_images_centered_generator(max_x, max_y, path):
            i = np.expand_dims(i, axis=0)
            i = i / 255
            yield (i,i)
    index = 0
    for i, target in centered_image_generator():
        plt.imsave(f'predictions/orig{index}.png',i[0])
        pred = model.predict(i)
        for p in pred:
            print("pred", p)
            plt.imsave(f'predictions/pred{index}.png',p)

        index += 1
        if index == 10:
            break

if __name__ == '__main__':
    train_on_images()
    load_model_and_predict()
    exit(0)
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

    def centered_image_generator():
        for i in load_images_centered():
            yield (i,i)


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
