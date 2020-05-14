import keras

import keras.layers as layers
from keras.models import Model
import numpy as np
from keras import backend as K
from keras import objectives
from functools import reduce
from keras.losses import mse, binary_crossentropy

import src.util.Arguments as Arguments


def from_argument_choice(choice: str, shape):
    if choice == Arguments.get_model_choice(Arguments.FC):
        model = autoencoder(shape, num_reductions=1)

    if choice == Arguments.get_model_choice(Arguments.CONV):
        # 4,2,64 decreasing, 4,2,16 increasing
        #model = conv_autoencoder(shape, num_reductions=4, filter_reduction_on=2, num_filters_start=8, increasing=True)
        model = conv_autoencoder(shape, num_reductions=6, filter_reduction_on=1, num_filters_start=8, increasing=True)


    if choice == Arguments.get_model_choice(Arguments.VAE):
        model, log_var, mu = vae_autoencoder(shape)

    if choice == Arguments.get_model_choice(Arguments.CONVVAE):
        model, encoder, decoder = conv_vae(shape)

    if choice == Arguments.get_model_choice(Arguments.FCS):
        model = autoencoder(shape, num_reductions=3)

    if choice == Arguments.get_model_choice(Arguments.FCSS):
        model = autoencoder(shape, num_reductions=6)

    return model


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

    l = e
    # l = layers.Flatten()(e)
    # l = layers.Dense(dense_size, activation='relu')(l)

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


def autoencoder(image_shape, num_reductions=1):
    image_x = image_shape[0]
    image_y = image_shape[1]
    image_z = image_shape[2]

    total_pixels = image_y * image_x * image_z

    inputs = keras.Input(shape=image_shape, name='image')
    x = layers.Flatten(name='flattened')(inputs)  # turn image to vector.

    current_size = total_pixels // 2
    for _ in range(num_reductions):
        x = layers.Dense(current_size, activation='relu', name=f'encoder_{_}')(x)
        current_size //= 2

    x = layers.Dense(current_size, activation='relu', name='middle_layer')(x)
    current_size *= 2

    for _ in range(num_reductions):
        x = layers.Dense(current_size, activation='relu', name=f'decoder_{_}')(x)
        current_size *= 2

    outputs = layers.Dense(total_pixels, activation='relu', name='reconstructed')(x) #sigmoid
    outputs = layers.Reshape(image_shape)(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    customAdam = keras.optimizers.Adam(lr=0.0001)

    model.compile(optimizer=customAdam,  # Optimizer
                  # Loss function to minimize
                  loss="mean_squared_error",
                  # List of metrics to monitor
                  metrics=["mean_squared_error"])
    model.summary()
    assert model.output_shape[1:] == image_shape, f'Output Shape {model.output_shape[1:]} is not equal input shape {image_shape}'
    return model


# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
# log var = sigma, standardavvik, mu = mean
def vae_loss(image_shape, log_var, mu):
    def custom_vae_loss(y_true, y_pred):

        # https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
        #xent_loss = objectives.binary_crossentropy(y_true, y_pred)
        xent_loss = objectives.mse(y_true, y_pred)
        kl_loss = - 0.5 * K.mean(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)


    return custom_vae_loss


def get_dummy_loss(image_shape):
    def dummy_loss(y_true, y_pred):
        xent_loss = objectives.binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        return K.mean(xent_loss)

    return dummy_loss


# https://github.com/lyeoni/keras-mnist-VAE/blob/master/keras-mnist-VAE.ipynb
def vae_autoencoder(image_shape):
    # network parameters
    image_shape_flat = reduce(lambda x, y: x * y, image_shape)
    batch_size = 1
    n_hidden = 512 # 512
    z_dim = 2
    print(image_shape)
    # encoder
    inputs = layers.Input(shape=image_shape)
    x = layers.Flatten(name='flattened')(inputs)  # turn image to vector.

    x_encoded = layers.Dense(n_hidden, activation='relu')(x)

    x_encoded = layers.Dense(n_hidden // 2, activation='relu')(x_encoded)

    mu = layers.Dense(z_dim, name='mu')(x_encoded)
    log_var = layers.Dense(z_dim, name='log')(x_encoded)

    # sampling function
    def sampling(args):
        mu, log_var = args
        eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
        return mu + K.exp(log_var) * eps

    z = layers.Lambda(sampling, output_shape=(z_dim,))([mu, log_var])

    # decoder
    z_decoded = layers.Dense(n_hidden // 2, activation='relu')(z)
    z_decoded = layers.Dense(n_hidden, activation='relu')(z_decoded)
    y = layers.Dense(image_shape_flat, activation='relu')(z_decoded)
    outputs = layers.Reshape(image_shape)(y)

    # build model
    vae = Model(inputs, outputs)
    # vae.add_loss(vae_loss)
    customAdam = keras.optimizers.Adam(lr=0.0001)
    vae.compile(optimizer=customAdam, loss=vae_loss(image_shape, log_var, mu))
    vae.summary()
    return vae, log_var, mu

def conv_vae(image_shape):
    verbosity = 1
    latent_dim = 2
    num_channels = 3

    i       = layers.Input(shape=image_shape, name='encoder_input')
    cx      = layers.Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(i)
    cx      = layers.BatchNormalization()(cx)
    cx      = layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
    cx      = layers.BatchNormalization()(cx)
    x       = layers.Flatten()(cx)
    x       = layers.Dense(20, activation='relu')(x)
    x       = layers.BatchNormalization()(x)
    mu      = layers.Dense(latent_dim, name='mu')(x) # latent mu
    sigma   = layers.Dense(latent_dim, name='log')(x) # latent sigma
    conv_shape = K.int_shape(cx)

    def sample_z(args):
        mu, sigma = args
        batch     = K.shape(mu)[0]
        dim       = K.int_shape(mu)[1]
        eps       = K.random_normal(shape=(batch, dim))
        return mu + K.exp(sigma / 2) * eps

    z       = layers.Lambda(sample_z, output_shape=(latent_dim, ), name='z')([mu, sigma])
    encoder = Model(i, [mu, sigma, z], name='encoder')
    encoder.summary()
    
    d_i   = layers.Input(shape=(latent_dim, ), name='decoder_input')
    x     = layers.Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(d_i)
    x     = layers.BatchNormalization()(x)
    x     = layers.Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
    cx    = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    cx    = layers.BatchNormalization()(cx)
    cx    = layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same',  activation='relu')(cx)
    cx    = layers.BatchNormalization()(cx)
    o     = layers.Conv2DTranspose(filters=num_channels, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)
    decoder = Model(d_i, o, name='decoder')
    decoder.summary()
    
    vae_outputs = decoder(encoder(i)[2])
    vae         = Model(i, vae_outputs, name='vae')
    vae.summary()
    vae.compile(optimizer='adam', loss=vae_loss(image_shape, mu, sigma))
    return vae, encoder, decoder
    


if __name__ == '__main__':
    shape = (64, 64, 3)

    ae = conv_autoencoder(shape, num_reductions=4, filter_reduction_on=2, num_filters_start=8, increasing=True)
