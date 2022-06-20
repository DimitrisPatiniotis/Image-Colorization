import sys
sys.path.insert(1, '../Utils/')
from settings import *
from dataloader import Dataloader
from models import model_main_with_mobilenet
import numpy as np

def train(model, l, ab, epochs, batch_size=32, training_size=5000):
    # Setup the training input data (grayscale images)
    model_X = l[:training_size]
    model_X = np.repeat(model_X[..., np.newaxis], 3, -1)
    model_Y = ab[:training_size]
    # Data Normalization
    model_X = (model_X.astype('float32') - 127.5) / 127.5
    model_Y = (model_Y.astype('float32') - 127.5) / 127.5

    history = model.fit(model_X, model_Y, epochs=epochs, validation_split=0.1, batch_size=batch_size)
    return history


def main():
    data_loader = Dataloader()
    data_loader.load()
    model = model_main_with_mobilenet()
    train(model, data_loader.x, data_loader.y, 10)


if __name__ == '__main__':
    main()