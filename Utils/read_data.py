import time
import numpy as np 
import os
from general_utils import *
from random import randint


def load_data(ds_dir='../Datasets/', print_time=True):
    if print_time:
        print('Loading Data...')
        start_time = time.time()
    ab1 = np.load(os.path.join(ds_dir + 'ab/ab1.npy'))
    ab2 = np.load(os.path.join(ds_dir + 'ab/ab2.npy'))
    ab3 = np.load(os.path.join(ds_dir + 'ab/ab3.npy'))
    ab = np.concatenate([ab1, ab2, ab3], axis=0)
    gray = np.load(os.path.join(ds_dir,"l/gray_scale.npy"))
    if print_time:
        print("Data loaded successfully in {} seconds".format(round(time.time() - start_time, 2)))
    return ab, gray

def get_colorized_images(ab, gray, print_time=True, print_sample=False):
    if print_time:
        print('Getting colorized images...')
        start_time = time.time()
    images = np.zeros((len(gray),224,224,3), dtype='uint8')
    for i in range(len(gray)):
        l = (gray[i]).reshape((224,224,1))
        a_b = (ab[i])
        images[i] = to_rgb_image(l, a_b)
    if print_time:
        print("Got {} colorized images successfully in {} seconds".format(len(gray), round(time.time() - start_time, 2)))
    if print_sample:
        display(images[0])
    return images

def display_image_from_dataset(ab, gray, index=randint(0, 25000)):
    l,a_b = gray[index].reshape((224,224,1)),ab[index]
    rgb_image = to_rgb_image(l,a_b)
    display(rgb_image)
    display(l[:,:,0])
    return rgb_image

if __name__ == '__main__':
    print('Data Loader Util')