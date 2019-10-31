import numpy as np, os

from time import time
from datetime import datetime

from keras import models, layers, optimizers, initializers, callbacks, regularizers
from keras.utils import multi_gpu_model, plot_model

class DCGANnetwork:
    def __init__(self):
        print('DCGAN network')

