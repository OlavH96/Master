import configparser
import os
from pathlib import Path

import pandas as pd
import tensorflow as tf

from object_detection.utils import dataset_util

root_dir = Path.cwd()

config = configparser.ConfigParser()
config.read(root_dir / 'config.ini')

size_x = config['Model']['size_generated_image_x']
size_y = config['Model']['size_generated_image_y']
config_name = config['Model']['config_name']
config_path = config['Model']['config_path']
training_steps = config['Model']['training_steps']
num_generated_images = config['Model']['num_generated_images']
template_path = config['Model']['template_path']

path_to_image_generation_data = config['Model']['path_to_image_generation_data']

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(path, filename, labels):
    # TODO(user): Populate the following variables from your example.
    height = size_y  # Image height
    width = size_x  # Image width
    bfilename = filename.encode()  # Filename of the image. Empty if image is not from file
    with tf.gfile.GFile(os.path.join(path, filename), 'rb') as f:
        encoded_image_data = f.read()  # Encoded image bytes
    image_format = b'jpeg'  # b'jpeg' or b'png'

    xmins = labels.xmin.tolist()  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = labels.xmax.tolist()  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = labels.ymin.tolist()  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = labels.ymax.tolist()  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = list(map(str.encode, labels.classlabel))  # List of string class name of bounding box (1 per box)
    classes = labels.classid.tolist()  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(bfilename),
        'image/source_id': dataset_util.bytes_feature(bfilename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):

    LABEL_FILE = f'{path_to_image_generation_data}/generated/train_labels.csv'
    IMAGES_PATH = f'{path_to_image_generation_data}/generated/images'
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    labels = pd.read_csv(LABEL_FILE)

    for filename in os.listdir(IMAGES_PATH):
        tf_example = create_tf_example(IMAGES_PATH, filename, labels[labels.filename == filename])
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
