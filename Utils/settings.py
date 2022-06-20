import numpy as np

BASE_DIR = '../'

IMG_RSZ_H = 224
IMG_RSZ_W = 224

def set_seed(num=50):
    np.random.seed(num)

if __name__ == '__main__':
    print('General Settings Util')