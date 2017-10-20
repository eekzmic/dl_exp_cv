import os
from glob import glob
from random import sample
import shutil

def make(train=True):
    suffix = 'train' if train else 'test'
    src_dir = '/srv/datasets/cifar10_pictures/' + suffix
    target_dir = 'mini_cifar/' + suffix
    size = 150 if train else 20

    directories = [p for p in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, p))]
    for directory in directories:
        this_directory = os.path.join(src_dir, directory)
        png_images = glob(os.path.join(this_directory, '*.png'))
        copy_images = sample(png_images, size)
        if not os.path.isdir(os.path.join(target_dir, directory)):
            os.mkdir(os.path.join(target_dir, directory))
        for copy_image in copy_images:
            shutil.copy2(copy_image, os.path.join(os.path.join(target_dir, directory),
                                                  os.path.basename(copy_image)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--train',action='store_true' , default=False)
    args = parser.parse_args()
    make(train=args.train)