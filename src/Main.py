import glob
import os

import keras
import tensorflow as tf

import keras.layers as layers
from keras.models import load_model
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from src.util.ImageLoader import load_images_generator, resize_image
import numpy as np
import logging as log
import random
from src.util.Arguments import anomaly_arguments
from keras import backend as K
from keras import objectives
from scipy.stats import norm
from functools import reduce
from PIL import Image
from src.train.Models import autoencoder, conv_autoencoder, vae_autoencoder, vae_loss, get_dummy_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def image_generator(path, max_x, max_y):
    for i in load_images_generator(path, color_mode='HSV'):
        i = resize_image(i, max_x, max_y)
        i = np.array(i)
        i = np.expand_dims(i, axis=0)
        i = i / 255
        yield (i, i)


def centered_image_generator(path, max_x, max_y):
    while True:
        for i, o in image_generator(path, max_x, max_y):
            yield (i, o)


def train_on_images(epochs, max_x, max_y, path, model_type, model_name):
    sess = tf.Session()
    keras.backend.set_session(sess)

    # max_x = max([i.shape[0] for i in images])
    # max_y = max([i.shape[1] for i in images])
    # max_x, max_y = find_max_min_image_size(path = 'detected_images/*.png')
    # print(max_x, max_y) # 304, 298

    epochs = epochs
    shape = (max_y, max_x, 3)

    if model_type == 'fully-connected':
        model = autoencoder(shape)
    if model_type == 'conv':
        # 4,2,64 decreasing, 4,2,16 increasing
        model = conv_autoencoder(shape, num_reductions=4, filter_reduction_on=2, num_filters_start=16, increasing=True)
    if model_type == 'vae':
        model, log_var, mu = vae_autoencoder(shape)
        print(log_var, mu)

    steps = len(glob.glob(path))

    # define the checkpoint
    filepath = model_name
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    log.info('Fitting model...')
    history = model.fit_generator(centered_image_generator(path, max_x, max_y), epochs=epochs, steps_per_epoch=steps,
                                  callbacks=callbacks_list)
    loss = history.history['loss']

    plt.plot(loss)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(f'training_loss_{model_name}.png')

    log.info('Finished fitting model')
    model.save(model_name)
    return model


def load_model_and_predict(model_path, num_predictions, path, max_x, max_y, model_type, model=None):
    # vae_loss(image_shape=(max_x, max_y, 3), log_var=0.5, mu=0.5) 
    im_shape = (max_x, max_y, 3)
    if model_type == 'vae' and not model:
        _, log_var, mu = vae_autoencoder(im_shape)
        model = load_model(model_path, custom_objects={'custom_vae_loss': vae_loss(im_shape, log_var, mu)})
        # model = load_model(model_path, custom_objects={'custom_vae_loss': lambda x,y: K.mean(np.array(0))})#vae_loss(im_shape, log_var, mu)})

        max_x = model.input_shape[1]
        max_y = model.input_shape[2]
        # o = model.get_layer('dense_2')
        # mu = model.get_layer('dense_3')(o)
        # log_var = model.get_layer('dense_4')(o)

        model = load_model(model_path, custom_objects={'custom_vae_loss': get_dummy_loss((max_x, max_y, 3))})
        # model = load_model(model_path, custom_objects={'custom_vae_loss': vae_loss((max_x, max_y, 3), mu, log_var)})

    if model_type != 'vae' and not model:
        model = load_model(model_path)
    model.summary()
    print("Loaded Model", model, model.input_shape)
    max_x = model.input_shape[1]
    max_y = model.input_shape[2]

    # create_manifold(model, max_x)
    # exit(1)
    images = list(image_generator(path, max_x, max_y))
    random.shuffle(images)
    index = 0
    print(f'Loaded {len(images)} images')
    for i, target in images:  # centered_image_generator(path, max_x, max_y):
        pred = model.predict(i)
        evaluate = model.evaluate(i, target)
        for ii in i:
            ii = Image.fromarray((ii * 255).astype(np.uint8), 'HSV')
            ii = ii.convert("RGB")
            ii = np.array(ii)
            plt.imsave(f'predictions/orig{index}.png', ii)

        print(evaluate)
        for p in pred:
            p = Image.fromarray((p * 255).astype(np.uint8), 'HSV')
            p = p.convert('RGB')
            p = np.array(p)
            plt.imsave(f'predictions/pred{index}_{str(evaluate)}.png', p, vmin=0, vmax=1)

        index += 1
        if index == num_predictions:
            break


def create_manifold(generator, max_x):
    n = 15  # figure with 15x15 digits
    digit_size = max_x
    figure = np.zeros((digit_size * n, digit_size * n))

    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            print(z_sample)
            z_sample = z_sample[0].reshape(digit_size, digit_size)
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imsave('sign_manifold.png', figure)
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


if __name__ == '__main__':

    args = anomaly_arguments()

    log.info('Arguments', args)
    print("Arguments", args)
    model = None
    if args.do_training:
        model = train_on_images(
            epochs=args.epochs,
            path=args.path,
            max_x=args.max_x,
            max_y=args.max_y,
            model_type=args.model_type,
            model_name=args.model
        )
    if args.do_predict:
        load_model_and_predict(
            model_path=args.model,
            num_predictions=args.num_predictions,
            max_x=args.max_x,
            max_y=args.max_y,
            path=args.path,
            model_type=args.model_type,
            model=model
        )
