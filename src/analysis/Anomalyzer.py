"""
Create "Anomaly" signs from normal images
TODO: Add arguments?
"""
import numpy as np

import src.util.ImageLoader as ImageLoader
import random
import matplotlib.pyplot as plt
import src.util.Filenames as Filenames

TEMPLATES = './analysis/resources/*'

def load_templates(path):
    images, names = ImageLoader.load_images(path)
    return images, names


def anomalyze_single_image(image, template):
    orig_x, orig_y = image.size
    image = ImageLoader.resize_image(image, 64, 64)
    image.paste(template, (0, 0), mask=template)
    image = ImageLoader.resize_image(image, orig_x, orig_y)
    return image


def anomalize(images: [], templates=TEMPLATES) -> []:
    templates, template_names = load_templates(templates)

    output = []

    for i in images:
        template, name = random.choice(list(zip(templates, template_names)))
        new_image = anomalyze_single_image(i, template)
        output.append(new_image)
    return output


if __name__ == '__main__':
    save_dir = './analysis/output'
    images, names = ImageLoader.load_images('./analysis/test/*')
    new_images = anomalize(images)

    for i, n in zip(new_images, names):
        n = Filenames.remove_path(n)
        n = ".".join(n.split('.')[:-1])
        plt.imsave(save_dir+"/"+n+"anomaly.png", np.array(i))

