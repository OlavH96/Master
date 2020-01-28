import glob
import os

import keras
import tensorflow as tf

from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import src.util.Files as Files
from src.util.ImageLoader import load_images_generator, resize_image, load_images_generator_with_filename
import numpy as np
import logging as log
import random
from src.util.Arguments import anomaly_arguments, get_model_choice
import src.util.Arguments as Arguments
from scipy.stats import norm
from PIL import Image
from src.train.Models import autoencoder, conv_autoencoder, vae_autoencoder, vae_loss, get_dummy_loss
import src.util.Filenames as Filenames

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def image_generator(path, max_x, max_y, color_mode="RGB"):
    for i in load_images_generator(path, color_mode=color_mode):
        i = resize_image(i, max_x, max_y)
        i = np.array(i)
        i = np.expand_dims(i, axis=0)
        i = i / 255
        yield (i, i)


def image_generator_with_filename(path, max_x, max_y, color_mode="RGB"):
    for i, f in load_images_generator_with_filename(path, color_mode=color_mode):
        i = resize_image(i, max_x, max_y)
        i = np.array(i)
        i = np.expand_dims(i, axis=0)
        i = i / 255
        yield (i, f)


def centered_image_generator(path, max_x, max_y, color_mode="RGB"):
    while True:
        for i, o in image_generator(path, max_x, max_y, color_mode=color_mode):
            yield (i, o)


def train_on_images(epochs, max_x, max_y, path, model_type, model_name, arg_steps, color_mode="RGB"):
    sess = tf.Session()
    keras.backend.set_session(sess)

    # max_x = max([i.shape[0] for i in images])
    # max_y = max([i.shape[1] for i in images])
    # max_x, max_y = find_max_min_image_size(path = 'detected_images/*.png')
    # print(max_x, max_y) # 304, 298

    epochs = epochs
    shape = (max_y, max_x, 3)
    if model_type == get_model_choice(Arguments.FC):
        model = autoencoder(shape, num_reductions=1)
    if model_type == get_model_choice(Arguments.CONV):
        # 4,2,64 decreasing, 4,2,16 increasing
        model = conv_autoencoder(shape, num_reductions=4, filter_reduction_on=2, num_filters_start=16, increasing=True)
    if model_type == get_model_choice(Arguments.VAE):
        model, log_var, mu = vae_autoencoder(shape)
        print(log_var, mu)
    if model_type == get_model_choice(Arguments.FCS):
        model = autoencoder(shape, num_reductions=4)

    steps = len(glob.glob(path))
    if arg_steps != 0:
        steps = arg_steps
    model.summary()

    # define the checkpoint
    checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    log.info('Fitting model...')
    history = model.fit_generator(centered_image_generator(path, max_x, max_y, color_mode=color_mode), epochs=epochs, steps_per_epoch=steps,
                                  callbacks=callbacks_list)
    model.save(model_name)
    loss = history.history['loss']
    try:
        plt.plot(loss)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(f'training_loss_{model_name}.png')
    except:
        log.info("Failed to create loss graph")

    log.info('Finished fitting model')
    return model


def load_model_and_predict(model_path, num_predictions, path, max_x, max_y, model_type, model=None, color_mode="RGB"):
    # vae_loss(image_shape=(max_x, max_y, 3), log_var=0.5, mu=0.5) 
    im_shape = (max_x, max_y, 3)
    if model_type == get_model_choice(Arguments.VAE) and not model:
        _, log_var, mu = vae_autoencoder(im_shape)
        model = load_model(model_path, custom_objects={'custom_vae_loss': vae_loss(im_shape, log_var, mu)})
        # model = load_model(model_path, custom_objects={'custom_vae_loss': lambda x,y: K.mean(np.array(0))})#vae_loss(im_shape, log_var, mu)})

        max_x = model.input_shape[1]
        max_y = model.input_shape[2]
        # o = model.get_layer('dense_2')
        # mu = model.get_layer('dense_3')(o)
        # log_var = model.get_layer('dense_4')(o)

        # model = load_model(model_path, custom_objects={'custom_vae_loss': get_dummy_loss((max_x, max_y, 3))})
        # model = load_model(model_path, custom_objects={'custom_vae_loss': vae_loss((max_x, max_y, 3), mu, log_var)})

    if model_type != get_model_choice(Arguments.VAE) and not model:
        model = load_model(model_path)
    model.summary()
    print("Loaded Model", model, model.input_shape)
    max_x = model.input_shape[1]
    max_y = model.input_shape[2]

    images = list(image_generator_with_filename(path, max_x, max_y, color_mode=color_mode))
    random.shuffle(images)
    index = 0
    print(f'Loaded {len(images)} images')

    model_name = model_path.split('.')[0]
    save_dir = Files.mkdir(f'./predictions/{model_name}')
    for i, filename in images:  # centered_image_generator(path, max_x, max_y):
        hashed = Filenames.md5hash(filename)
        pred = model.predict(i)
        evaluate = model.evaluate(i, i)
        for ii in i:
            if color_mode == 'HSV':
                ii = Image.fromarray((ii * 255).astype(np.uint8), 'HSV')
                ii = ii.convert("RGB")
                ii = np.array(ii)
            plt.imsave(str(save_dir / f'orig_{model_path}_{hashed}_{index}.png'), ii)

        if type(evaluate) is list:
            evaluate = evaluate[0]
        print(index, evaluate)

        for p in pred:
            if color_mode == 'HSV':
                p = Image.fromarray((p * 255).astype(np.uint8), 'HSV')
                p = p.convert('RGB')
                p = np.array(p)
            plt.imsave(str(save_dir / f'pred_{model_path}_{index}_{hashed}_{str(evaluate)}.png'), p, vmin=0, vmax=1)

        index += 1
        if index == num_predictions:
            break


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
            model_name=args.model,
            arg_steps=args.steps,
            color_mode=args.color
        )
    if args.do_predict:
        load_model_and_predict(
            model_path=args.model,
            num_predictions=args.num_predictions,
            max_x=args.max_x,
            max_y=args.max_y,
            path=args.path,
            model_type=args.model_type,
            model=model,
            color_mode=args.color
        )
