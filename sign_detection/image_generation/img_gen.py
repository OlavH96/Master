import configparser
import os
import random
import time

import numpy as np
from multiprocessing import Process, Queue, cpu_count
from PIL import Image, ImageFilter, ImageOps
import pathlib

root_dir = pathlib.Path.cwd()

config = configparser.ConfigParser()
config.read(root_dir / 'config.ini')

num_generated_images = config['Model']['num_generated_images']
output_size_x        = config['Model']['size_generated_image_x']
output_size_y        = config['Model']['size_generated_image_y']
img_gen_dir          = config['Model']['path_to_image_generation_data']

output_image_size = (output_size_x, output_size_y)


class Background:
    BACKGROUND_PATH = os.path.join(img_gen_dir, 'backgrounds')

    def __init__(self, size=None):
        if size is None:
            size = output_image_size
        background_images = os.listdir(self.BACKGROUND_PATH)
        self.size = size
        self.backgrounds = [Image.open(os.path.join(self.BACKGROUND_PATH, image)).convert("RGBA").resize(self.size)
                            for image in background_images
                            if not image.startswith('.')]

    def get(self):
        return random.choice(self.backgrounds).copy()


class Sign:
    SIGN_PATH = os.path.join(img_gen_dir, 'signs/png')

    def __init__(self):
        sign_paths = sorted(os.listdir(self.SIGN_PATH))
        self.signs = []
        for folder in sign_paths:
            if folder.startswith('.'):
                continue
            sign_images = os.listdir(os.path.join(self.SIGN_PATH, folder))
            images = [Image.open(os.path.join(self.SIGN_PATH, folder, image)).convert("RGBA") for image in sign_images]
            self.signs.append((folder, images))

    def get(self):
        i = random.randrange(0, len(self.signs))
        return i + 1, self.signs[i][0], random.choice(self.signs[i][1])


class Generator(Process):
    OUTPUT_PATH = os.path.join(img_gen_dir, 'generated/images')

    def __init__(self, tasks, results, rotation=(-20, 20), scale=(0.08, 0.25), signs=(2, 5), noise=(-10, 10),
                 stretch=(0.6, 1.4),
                 fade=(0.6, 1.4)):
        super().__init__()
        self.tasks = tasks
        self.results = results
        self.rotation = rotation
        self.scale = scale
        self.signs = signs
        self.noise = noise
        self.stretch = stretch
        self.fade = fade

    def run(self):
        self.background = Background()
        self.sign = Sign()

        while not self.tasks.empty():
            task = self.tasks.get()
            fname = '%06d.jpg' % task
            image, data = self.generate_image(fname)
            image.convert('RGB').save(os.path.join(self.OUTPUT_PATH, fname))
            self.results.put((task, data))

    def generate_image(self, fname):
        background = self.background.get()
        background = self.img_noise(background)
        data_format = '%s,%r,%r,%r,%s,%r,%r,%r,%r\n'
        data = ''
        for i in range(random.randint(*self.signs)):
            sign_id, sign_label, im = self.sign.get()
            im = self.img_fade_color(im)
            im = self.img_stretch(im)
            im = self.img_rotate(im)
            im = self.img_scale(background, im)
            im = self.img_blur(im)
            im = self.img_noise(im)
            pos_x, pos_y = self.img_paste(background, im)
            minmax = (
                pos_x / background.size[0],
                pos_y / background.size[1],
                (pos_x + im.size[0]) / background.size[0],
                (pos_y + im.size[1]) / background.size[1],
            )
            data += data_format % (fname, *self.background.size, sign_id, sign_label, *minmax)
        return background, data

    def img_paste(self, background, im):
        pos_x = random.randrange(0, background.size[0] - im.size[0])
        pos_y = random.randrange(0, background.size[1] - im.size[1])
        background.paste(im, (pos_x, pos_y), mask=im)
        return pos_x, pos_y

    def img_rotate(self, im):
        return im.rotate(random.randrange(*self.rotation), expand=True)

    def img_scale(self, background, im):
        current_scale = max([s / b for s, b in zip(im.size, background.size)])
        scale = (self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()) / current_scale
        return im.resize((int(scale * im.size[0]), int(scale * im.size[1])))

    def img_blur(self, im, r=1):
        return im.filter(ImageFilter.GaussianBlur(r))

    def img_detail(self, im):
        return im.filter(ImageFilter.DETAIL)

    def img_noise(self, im):
        img = np.asarray(im)
        noise = np.random.uniform(*self.noise, img.shape)
        noise[:, :, 3] = 0
        new_img = np.uint8(np.clip(img + noise, 0, 255))
        return Image.fromarray(new_img, 'RGBA')

    def img_pad(self, im, amount):
        left = right = int(im.size[0] * amount)
        top = bottom = int(im.size[1] * amount)
        return ImageOps.expand(im, (left, top, right, bottom))

    def img_stretch(self, im):
        new_width = int((self.stretch[0] + (self.stretch[1] - self.stretch[0]) * random.random()) * im.size[0])
        return im.resize((new_width, im.size[1]))

    def img_fade_color(self, im):
        img = np.asarray(im)
        fade = [self.fade[0] + (self.fade[1] - self.fade[0]) * np.random.random() for _ in range(3)]
        fade.append(1)
        fade = np.reshape([fade] * (im.size[0] * im.size[1]), img.shape)
        new_img = np.uint8(np.clip(img * fade, 0, 255))
        return Image.fromarray(new_img, 'RGBA')


def generate(count):
    OUTPUT_PATH = os.path.join(img_gen_dir, 'generated')

    if not pathlib.Path(OUTPUT_PATH).exists():
        os.mkdir(OUTPUT_PATH)
        os.mkdir('%s/images' % OUTPUT_PATH)

    tasks = Queue()
    results = Queue()

    for i in range(count):
        tasks.put(i)

    workers = []
    for i in range(int(cpu_count())):
        workers.append(Generator(tasks, results))

    for worker in workers:
        worker.start()

    with open(os.path.join(OUTPUT_PATH, 'train_labels.csv'), 'w') as f:
        f.write('filename,width,height,classid,classlabel,xmin,ymin,xmax,ymax\n')

    generated = 0
    while generated < count:
        if results.empty():
            time.sleep(1)
        else:
            generated += 1
            l = str(len(str(count)))
            print(('%0' + l + 'd/%0' + l + 'd') % (generated, count))
            task, data = results.get()
            with open(os.path.join(OUTPUT_PATH, 'train_labels.csv'), 'a') as f:
                f.write(data)


if __name__ == '__main__':
    generate(num_generated_images)
