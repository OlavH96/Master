import glob
import os

import keras
import tensorflow as tf

import keras.layers as layers
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Conv2D, Input, Reshape, Flatten, Conv2DTranspose, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from util.ImageLoader import load_images_generator, resize_image
import numpy as np
import logging as log
import random
from util.Arguments import anomaly_arguments

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# https://medium.com/analytics-vidhya/building-a-convolutional-autoencoder-using-keras-using-conv2dtranspose-ca403c8d144e
def conv_autoencoder(io_shape, num_reductions=5, filter_reduction_on=2, num_filters_start=32, increasing=True):
    inp = Input(io_shape)
    e = inp

    num_filters = num_filters_start
    for i, _ in enumerate(range(num_reductions)):
        e = Conv2D(num_filters, (3, 3), activation='relu', padding="same")(e)
        e = MaxPooling2D((2, 2), strides=2, padding="same")(e)

        if i % filter_reduction_on == 0:
            if increasing:
                num_filters = int(num_filters * 2)
            else:
                num_filters = int(num_filters / 2)

    e = Conv2D(num_filters, (3, 3), activation='relu', padding="same")(e)

    s = [int(i) for i in e.shape[1:]]
    # s[-1] = 1
    prod_s = np.prod(s)
    dense_size = prod_s
    reshape_shape = s

    l = Flatten()(e)
    l = Dense(dense_size, activation='relu')(l)

    # DECODER
    d = Reshape(reshape_shape)(l)

    for i, _ in enumerate(range(num_reductions)):
        d = Conv2DTranspose(num_filters, (3, 3), strides=2, activation='relu', padding='same')(d)

        if i % filter_reduction_on == 0:
            if increasing:
                num_filters = int(num_filters / 2)
            else:
                num_filters = int(num_filters * 2)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(d)
    ae = Model(inp, decoded)
    customAdam = keras.optimizers.Adam(lr=0.01, amsgrad=True)
    ae.compile(optimizer=customAdam, loss="mse")
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

    outputs = layers.Dense(total_pixels, activation='sigmoid', name='reconstructed_cat')(x)
    outputs = layers.Reshape(image_shape)(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    customAdam = keras.optimizers.Adam(lr=0.001)  # you have no idea how many times I changed this number

    model.compile(optimizer=customAdam,  # Optimizer
                  # Loss function to minimize
                  loss="mean_squared_error",
                  # List of metrics to monitor
                  metrics=["mean_squared_error"])
    model.summary()
    return model


def image_generator(path, max_x, max_y):
    for i in load_images_generator(path):
        i = resize_image(i, max_x, max_y)
        i = np.array(i)
        i = np.expand_dims(i, axis=0)
        i = i / 255
        yield (i, i)


def centered_image_generator(path, max_x, max_y):
    while True:
        for i, o in image_generator(path, max_x, max_y):
            yield (i, o)


def train_on_images(epochs, max_x, max_y, path):
    sess = tf.Session()
    keras.backend.set_session(sess)

    # max_x = max([i.shape[0] for i in images])
    # max_y = max([i.shape[1] for i in images])
    # max_x, max_y = find_max_min_image_size(path = 'detected_images/*.png')
    # print(max_x, max_y) # 304, 298

    epochs = epochs
    shape = (max_y, max_x, 3)
    # 4,2,64 decreasing, 4,2,16 increasing
    # model = conv_autoencoder(shape, num_reductions=4, filter_reduction_on=2, num_filters_start=16, increasing=True)#autoencoder(shape)
    model = autoencoder(shape)
    steps = len(glob.glob(path))

    # define the checkpoint
    filepath = "model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    log.info('Fitting model...')
    history = model.fit_generator(centered_image_generator(path, max_x, max_y), epochs=epochs, steps_per_epoch=steps,
                                  callbacks=callbacks_list)
    loss = history.history['loss']
    plt.plot(loss)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig("training_loss.png")

    log.info('Finished fitting model')
    model.save("autoencoder.h5")


def load_model_and_predict(model_path, num_predictions, path, max_x, max_y):
    model = load_model(model_path)
    model.summary()
    images = list(image_generator(path, max_x, max_y))
    random.shuffle(images)
    index = 0
    for i, target in images:  # centered_image_generator(path, max_x, max_y):
        plt.imsave(f'predictions/orig{index}.png', i[0])
        pred = model.predict(i)
        evaluate = model.evaluate(i, target)
        print(evaluate)
        for p in pred:
            plt.imsave(f'predictions/pred{index}.png', p, vmin=0, vmax=1)

        index += 1
        if index == num_predictions:
            break


if __name__ == '__main__':

    args = anomaly_arguments()
    log.info('Arguments', args)

    if args.do_training:
        train_on_images(
            epochs=args.epochs,
            path=args.path,
            max_x=args.max_x,
            max_y=args.max_y
        )
    if args.do_predict:
        load_model_and_predict(
            model_path=args.model,
            num_predictions=args.num_predictions,
            max_x=args.max_x,
            max_y=args.max_y,
            path=args.path
        )
