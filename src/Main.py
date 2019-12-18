import glob
import os

import keras
import tensorflow as tf

import keras.layers as layers
from keras.models import load_model
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from ImageLoader import load_images_generator, resize_image
import numpy as np
import logging as log
import random
from Arguments import anomaly_arguments
from keras import backend as K
from keras import objectives
from scipy.stats import norm
from functools import reduce
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# https://medium.com/analytics-vidhya/building-a-convolutional-autoencoder-using-keras-using-conv2dtranspose-ca403c8d144e
def conv_autoencoder(io_shape, num_reductions=5, filter_reduction_on=2, num_filters_start=32, increasing=True):
    inp = layers.Input(io_shape)
    e = inp

    num_filters = num_filters_start
    for i, _ in enumerate(range(num_reductions)):
        e = layers.Conv2D(num_filters, (3, 3), activation='relu', padding="same")(e)
        e = layers.MaxPooling2D((2, 2), strides=2, padding="same")(e)

        if i % filter_reduction_on == 0:
            if increasing:
                num_filters = int(num_filters * 2)
            else:
                num_filters = int(num_filters / 2)

    e = layers.Conv2D(num_filters, (3, 3), activation='relu', padding="same")(e)

    s = [int(i) for i in e.shape[1:]]
    # s[-1] = 1
    prod_s = np.prod(s)
    dense_size = prod_s
    reshape_shape = s

    l = layers.Flatten()(e)
    l = layers.Dense(dense_size, activation='relu')(l)

    # DECODER
    d = layers.Reshape(reshape_shape)(l)

    for i, _ in enumerate(range(num_reductions)):
        d = layers.Conv2DTranspose(num_filters, (3, 3), strides=2, activation='relu', padding='same')(d)

        if i % filter_reduction_on == 0:
            if increasing:
                num_filters = int(num_filters / 2)
            else:
                num_filters = int(num_filters * 2)

    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(d)
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

    inputs = keras.Input(shape=image_shape, name='image')
    x = layers.Flatten(name='flattened')(inputs)  # turn image to vector.

    x = layers.Dense(translator_layer_size, activation='relu', name='encoder')(x)
    x = layers.Dense(middle_layer_size, activation='relu', name='middle_layer')(x)
    x = layers.Dense(translator_layer_size, activation='relu', name='decoder')(x)

    outputs = layers.Dense(total_pixels, activation='sigmoid', name='reconstructed')(x)
    outputs = layers.Reshape(image_shape)(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    customAdam = keras.optimizers.Adam(lr=0.0001)

    model.compile(optimizer=customAdam,  # Optimizer
                  # Loss function to minimize
                  loss="mean_squared_error",
                  # List of metrics to monitor
                  metrics=["mean_squared_error"])
    model.summary()
    return model

def vae_loss(image_shape, log_var, mu):
    print('\n\n\n\n\n', type(log_var), type(mu))

    def custom_vae_loss(y_true, y_pred):
        print("VAE Loss", y_true.shape, y_pred.shape, log_var, mu)
        xent_loss = image_shape[0]* image_shape[1]* objectives.binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        kl_loss = - 0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        return vae_loss

    return custom_vae_loss 
def get_dummy_loss(image_shape):
    def dummy_loss(y_true, y_pred):
        
        xent_loss = image_shape[0]* image_shape[1]*objectives.binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        return K.mean(xent_loss)
    return dummy_loss

# https://github.com/lyeoni/keras-mnist-VAE/blob/master/keras-mnist-VAE.ipynb
def vae_autoencoder(image_shape):
    # network parameters
    image_shape_flat = reduce(lambda x, y: x * y, image_shape)
    batch_size, n_epoch = 1, 10
    n_hidden, z_dim = 256, 2
    print(image_shape)
    # encoder
    inputs  = layers.Input(shape=image_shape)
    x = layers.Flatten(name='flattened')(inputs)  # turn image to vector.

    x_encoded = layers.Dense(n_hidden, activation='relu')(x)

    x_encoded = layers.Dense(n_hidden//2, activation='relu')(x_encoded)
    
    mu = layers.Dense(z_dim)(x_encoded)
    log_var = layers.Dense(z_dim)(x_encoded)    

    # sampling function
    def sampling(args):
        mu, log_var = args
        eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
        return mu + K.exp(log_var) * eps

    z = layers.Lambda(sampling, output_shape=(z_dim,))([mu, log_var])

    # decoder
    z_decoded = layers.Dense(n_hidden//2, activation='relu')(z)
    z_decoded = layers.Dense(n_hidden, activation='relu')(z_decoded)
    y = layers.Dense(image_shape_flat, activation='sigmoid')(z_decoded)
    outputs = layers.Reshape(image_shape)(y)

    # loss
    #reconstruction_loss = objectives.binary_crossentropy(x, y) * image_shape_flat
    #kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis = -1)
    #vae_loss = reconstruction_loss + kl_loss

    #def my_vae_loss(y_true, y_pred):
    #    print("VAE Loss", y_true.shape, y_pred.shape)
    #    xent_loss = image_shape[0]* image_shape[1]* objectives.binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
    #    kl_loss = - 0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
    #    vae_loss = K.mean(xent_loss + kl_loss)
    #    return vae_loss
    
    # build model
    vae = Model(inputs, outputs)
    #vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop', loss=vae_loss(image_shape, log_var, mu))
    vae.summary()
    return vae, log_var, mu

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
    history = model.fit_generator(centered_image_generator(path, max_x, max_y), epochs=epochs, steps_per_epoch=steps, callbacks=callbacks_list)
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
        #model = load_model(model_path, custom_objects={'custom_vae_loss': lambda x,y: K.mean(np.array(0))})#vae_loss(im_shape, log_var, mu)})

        max_x = model.input_shape[1]
        max_y = model.input_shape[2]
        #o = model.get_layer('dense_2')
        #mu = model.get_layer('dense_3')(o)
        #log_var = model.get_layer('dense_4')(o)
        
        model = load_model(model_path, custom_objects={'custom_vae_loss': get_dummy_loss((max_x, max_y, 3))})
        #model = load_model(model_path, custom_objects={'custom_vae_loss': vae_loss((max_x, max_y, 3), mu, log_var)})

    if model_type != 'vae' and not model:
        model = load_model(model_path)
    model.summary()
    print("Loaded Model", model, model.input_shape)
    max_x = model.input_shape[1]
    max_y = model.input_shape[2]

    #create_manifold(model, max_x)
    #exit(1)
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

    n = 15 # figure with 15x15 digits
    digit_size = max_x 
    figure = np.zeros((digit_size * n, digit_size * n))
    
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n)) 
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            print(z_sample)
            z_sample= z_sample[0].reshape(digit_size, digit_size)
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

    #i = Image.open('training_loss.png')
    #orig = i
    #i = i.convert('HSV')
    #i = np.array(i)
    #i = i / 255
    #i = (i * 255).astype('uint8')
    #i = Image.fromarray(i, 'HSV')
    #i = i.convert("RGB")
    #i.save('training_loss_rgb.png', "PNG")

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
            model_name = args.model
        )
    if args.do_predict:
        load_model_and_predict(
            model_path=args.model,
            num_predictions=args.num_predictions,
            max_x=args.max_x,
            max_y=args.max_y,
            path=args.path,
            model_type=args.model_type,
            model = model
        )
