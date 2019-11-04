import numpy as np, os, configparser

from time import time
from datetime import datetime

from keras.layers import Dense, Dropout, Flatten, Reshape, BatchNormalization, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.losses import binary_crossentropy
from keras.initializers import RandomNormal

from keras import models, optimizers, initializers, callbacks, regularizers
from keras.utils import multi_gpu_model, plot_model

from config.net_config import NetworkConfig

class DCGANnetwork:
    def __init__(self, PATH_CONFIG='./config/', PATH_OUTPUT=None):
        print('DCGAN network')

        # Configure networks
        self.path_config = PATH_CONFIG
        self.conf = NetworkConfig(self.path_config+'gan.ini')
        self.optim = optimizers.Adam(lr=self.conf.lr, beta_1=self.conf.beta1)
        self.loss = 'binary_crossentropy'

        # Create output directory and sub-directories
        if PATH_OUTPUT != None:
            self.path_output = PATH_OUTPUT + datetime.now().strftime('%d-%mT%H-%M-%S')
            if not os.path.exists(self.path_output):
                os.makedirs(self.path_output)
                os.makedirs(self.path_output+'/model')
                os.makedirs(self.path_output+'/checkpoints')
                os.makedirs(self.path_output+'/checkpoints/weights')
        else:
            self.path_output = datetime.now().strftime('%d-%mT%H-%M-%S')
            if not os.path.exists(self.path_output):
                os.makedirs(self.path_output)
                os.makedirs(self.path_output+'/model')
                os.makedirs(self.path_output+'/checkpoints')
                os.makedirs(self.path_output+'/checkpoints/weights')
        self.path_output += '/'


    def Generator(self):
        print('Create Generator network...')
        
        G = models.Sequential()

        # first block
        G.add(Dense(np.prod(self.conf.coarse_dim), 
                    input_dim=self.conf.input_dim,
                    kernel_initializer=initializers.RandomNormal(stddev=0.02), 
                    name='fully_connected_layer'))
        G.add(BatchNormalization(name='normalization_1'))
        G.add(LeakyReLU(alpha=self.conf.alpha))

        G.add(Reshape(self.conf.coarse_dim, name='low_resolution_output'))
        assert all(G.output_shape[1:] == self.conf.coarse_dim)

        # second block
        G.add(Conv2DTranspose(int(self.conf.coarse_dim[2]/2), 
                              kernel_size=self.conf.kernel_size, 
                              strides=1, 
                              padding='same', 
                              use_bias=False, 
                              name='convolve_coarse_input'))
        #print(G.output_shape[1:])
        G.add(BatchNormalization(name='normalization_2'))
        G.add(LeakyReLU(alpha=self.conf.alpha))            # alpha: slope coefficient
        
        # third block
        G.add(Conv2DTranspose(int(self.conf.coarse_dim[2]/4), 
                              kernel_size=self.conf.kernel_size, 
                              strides=self.conf.upsampl_size, 
                              padding='same', 
                              use_bias=False, 
                              name='convolve_upsampling_1'))
        #print(G.output_shape[1:])
        G.add(BatchNormalization(name='normalization_3'))
        G.add(LeakyReLU(alpha=self.conf.alpha))
        
        # outro block
        G.add(Conv2DTranspose(int(self.conf.output_dim[2]), 
                              kernel_size=self.conf.kernel_size, 
                              strides=self.conf.upsampl_size, 
                              activation='tanh',
                              padding='same', 
                              use_bias=False, 
                              name='generated_output_picture'))
        #print(G.output_shape[1:])
        assert all(G.output_shape[1:] == self.conf.output_dim)

        # compile model
        G.compile(loss=self.loss, optimizers=self.optim)

        # Save model visualization
        plot_model(G, to_file=self.path_output+'model/generator_visualization.png', show_shapes=True, show_layer_names=True)

        return G

    def Adversary(self):
        print('Create Adversary network...')

        A = models.Sequential()

        # first block
        A.add(Conv2D(int(self.conf.coarse_dim[2]/4), 
                     input_shape=self.conf.input_dim, 
                     kernel_size=self.conf.kernel_size, 
                     strides=self.conf.upsampl_size,
                     kernel_initializer=initializers.RandomNormal(stddev=0.02),
                     activation=LeakyReLU(alpha=self.conf.alpha),
                     padding='same'))
        A.add(Dropout(self.conf.dropout))

        # second block
        A.add(Conv2D(int(self.conf.coarse_dim[2]/2),
                     input_shape=self.conf.input_dim, 
                     kernel_size=self.conf.kernel_size, 
                     strides=self.conf.upsampl_size, 
                     activation=LeakyReLU(alpha=self.conf.alpha),
                     padding='same'))
        A.add(Dropout(self.conf.dropout))

        # outro block
        A.add(Flatten())
        A.add(Dense(self.conf.output_dim[2], activation='sigmoid'))
        
        # compile model
        A.compile(loss=self.loss, optimizers=self.optim)
        A.trainable = False

        # Save model visualization
        plot_model(A, to_file=self.path_output+'model/adversary_visualization.png', show_shapes=True, show_layer_names=True)

        return A

    def GAN(self):
        generator = self.Generator()
        adversary = self.Adversary()
        
        # define input shape
        inputGAN = Input(shape=(self.conf.input_dim,))
        
        # Create GAN network by merging the generator and adversary network
        self.gan = models.Model(inputs=inputGAN, outputs=adversary(generator(inputGAN)))
        
        # compile network
        self.gan.compile(loss=self.loss, optimizer=self.optim)
        
        return self.gan