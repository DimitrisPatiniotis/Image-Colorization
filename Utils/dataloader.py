from numpy import disp
from read_data import *
from general_utils import *
from sklearn.utils import shuffle

class Dataloader():
    def __init__(self, test_size=0.3, show_info=True):
        self.test_size = test_size
        self.show_info = show_info
        self.x = None
        self.y = None

    def load(self):
        ab, gray = load_data(print_time=self.show_info)
        # Collecting and shuffling gray and colorized images
        self.x = gray
        self.y = ab
        self.x, self.y = shuffle(self.x, self.y, random_state=0)


if __name__ == '__main__':
    print('Data Loader Util')