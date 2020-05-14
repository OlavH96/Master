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
from src.train.Models import autoencoder, conv_autoencoder, vae_autoencoder, vae_loss, get_dummy_loss, from_argument_choice
import src.train.Models as Models
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


def train_on_images(epochs, max_x, max_y, path, model_type, model_name, arg_steps, validation_path, color_mode="RGB"):
    sess = tf.Session()
    keras.backend.set_session(sess)

    # max_x = max([i.shape[0] for i in images])
    # max_y = max([i.shape[1] for i in images])
    # max_x, max_y = find_max_min_image_size(path = 'detected_images/*.png')
    # print(max_x, max_y) # 304, 298

    epochs = epochs
    shape = (max_y, max_x, 3)
    model = Models.from_argument_choice(model_type, shape)

    steps = len(glob.glob(path))
    if arg_steps != 0:
        steps = arg_steps
    model.summary()

    # define the checkpoint
    checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    log.info('Fitting model...')
    if validation_path:
        history = model.fit_generator(generator=centered_image_generator(path, max_x, max_y, color_mode=color_mode),
                                      validation_data=centered_image_generator(validation_path, max_x, max_y, color_mode=color_mode),
                                      validation_steps=100, 
                                      epochs=epochs,
                                      steps_per_epoch=steps,
                                      callbacks=callbacks_list)
    else:
        history = model.fit_generator(generator=centered_image_generator(path, max_x, max_y, color_mode=color_mode),
                                      epochs=epochs,
                                      steps_per_epoch=steps,
                                      callbacks=callbacks_list)

    model.save(model_name)
    loss = history.history['loss']
    try:
        plt.plot(loss)
        if validation_path:
            val_loss = history.history['val_loss']
            plt.plot(val_loss, color='g')
        plt.title(model_name)
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

        model = load_model(model_path, compile=False)#custom_objects={'custom_vae_loss': vae_loss(im_shape, log_var, mu)})
        mu = model.get_layer('mu').output
        log_var = model.get_layer('log').output
        model.summary()
        print(mu, log_var)
        model.compile(optimizer='rmsprop', loss=vae_loss(im_shape, log_var, mu))

    if model_type == get_model_choice(Arguments.CONVVAE) and not model:

        model = load_model(model_path, compile=False)#custom_objects={'custom_vae_loss': vae_loss(im_shape, log_var, mu)})
        
        encoder  = model.get_layer('encoder')
        decoder  = model.get_layer('decoder')

        mu = encoder.get_layer('mu').output
        log_var = encoder.get_layer('log').output

        model.compile(optimizer='adam', loss=vae_loss(im_shape, log_var, mu))

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
    save_dir = Files.makedir_else_cleardir(f'./predictions/{model_name}_{Filenames.remove_path(Filenames.strip_path_modifier(path))}')

    for i, filename in images:  # centered_image_generator(path, max_x, max_y):
        hashed = Filenames.md5hash(filename)
        anomaly = "anomaly" in filename
        extra = "_anomaly_" if anomaly else "_normal_"
        pred = model.predict(i)
        print(pred.shape)
        
        for ii in i:
            if color_mode == 'HSV':
                ii = Image.fromarray((ii * 255).astype(np.uint8), 'HSV')
                ii = ii.convert("RGB")
                ii = np.array(ii)
            plt.imsave(str(save_dir / f'orig{extra}{hashed}_{index}.png'), ii)

        plt.imsave(str(save_dir / f'temp.png'), pred[0], vmin=0, vmax=1)
        print("input shape",i.shape)
        evaluate = model.evaluate(i, i)
        if type(evaluate) is list:
            evaluate = evaluate[0]
        print(index, evaluate)

        for p in pred:
            if color_mode == 'HSV':
                p = Image.fromarray((p * 255).astype(np.uint8), 'HSV')
                p = p.convert('RGB')
                p = np.array(p)
            plt.imsave(str(save_dir / f'pred{extra}{index}_{hashed}_{str(evaluate)}.png'), p, vmin=0, vmax=1)

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
            color_mode=args.color,
            validation_path=args.validation_path
        )
    if args.do_predict:
        load_model_and_predict(
            model_path=args.model,
            num_predictions=args.num_predictions,
            max_x=args.max_x,
            max_y=args.max_y,
            path=args.pred_path if args.pred_path else args.path,
            model_type=args.model_type,
            model=model,
            color_mode=args.color
        )
