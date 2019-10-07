import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import util.ImageLoader as ImageLoader
import Main as Main
import tensorflow as tf
import keras

# def train_model():
#
#     config = tf.ConfigProto(device_count={'CPU': 2})
#     sess = tf.Session(config=config)
#     keras.backend.set_session(sess)
#
#     model = Main.autoencoder(image_shape=images.shape[1:])
#     print(model)
#
#     model.fit(images, images, epochs=1)
#     prediction = model.predict(test_image)
#
#     plt.imshow(prediction)
#     plt.show()

if __name__ == '__main__':

    images = ImageLoader.load_images_centered()

    test_image = '../data/test.png'
    test_image = Image.open(test_image)
    test_image = ImageLoader.center_image_with_padding(test_image, *images.shape[1:3])
    test_image = np.array(test_image)
    print(test_image.shape)
    plt.imshow(test_image)
    plt.show()

    for i in images:
        plt.imshow(i)
        plt.show()

