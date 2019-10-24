import configparser
import os
from pathlib import Path

from label_gen import get_labels

root_dir = Path.cwd()

config = configparser.ConfigParser()
config.read(root_dir / 'config.ini')

size_x                  = config['Model']['size_generated_image_x']
size_y                  = config['Model']['size_generated_image_y']
config_name             = config['Model']['config_name']
config_path             = config['Model']['config_path']
training_steps          = config['Model']['training_steps']
num_generated_images    = config['Model']['num_generated_images']
template_path           = config['Model']['template_path']


def gen_config():
    CONFIG_NAME = config_name
    TEMPLATE_PATH = template_path
    CONFIG_PATH = config_path
    with open('%s.template' % os.path.join(TEMPLATE_PATH, CONFIG_NAME), 'r') as f:
        template = f.read()
    with open('%s' % os.path.join(CONFIG_PATH, CONFIG_NAME), 'w') as f:
        f.write(template % {
            'IMAGE_WIDTH': size_x,
            'IMAGE_HEIGHT': size_y,
            'IMAGE_COUNT': num_generated_images,
            'CLASS_COUNT': len(get_labels()),
            'TRAIN_STEPS': training_steps,
        })


if __name__ == '__main__':
    gen_config()
