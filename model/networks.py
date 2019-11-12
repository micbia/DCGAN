import numpy as np

from keras.layers import Dense, Dropout, Flatten, Reshape, BatchNormalization, Input, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.utils import multi_gpu_model, plot_model
from keras import models, optimizers, initializers, callbacks, regularizers

from config.net_config import NetworkConfig
from utils.other_utils import GenerateNoise, GenerateLabels, BatchSample, ScaleData

class NetworkComponents:
    def __init__(self, CONFIG_FILE, PATH_OUTPUT):
        
        # Configure networks
        self.conf = NetworkConfig(CONFIG_FILE)
        self.config_file = CONFIG_FILE
        self.path_output = PATH_OUTPUT
        self.optimizer = optimizers.Adam(lr=2e-4, beta_1=0.5)   # see Radford et al. 2015

        if(self.conf.type_of_gan == 'DC'):
            print('DC-GAN network')
            self.lossG = 'binary_crossentropy'
            self.lossA = 'binary_crossentropy'
            self.lossGAN = 'binary_crossentropy'
        elif(self.conf.type_of_gan == 'LS'):
            print('LS-GAN network')
            self.lossG = 'binary_crossentropy'
            self.lossA = 'mse'
            self.lossGAN = 'mse'
        elif(self.conf.type_of_gan == 'WGP'):
            print('WGAN-GP network')
        

    def Generator(self):
        print('Create Generator network...')
        G = models.Sequential()

        kinit = initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        # first block
        G.add(Dense(np.prod(self.conf.coarse_dim), 
                    input_dim=self.conf.input_dim,
                    kernel_initializer=kinit, 
                    name='fully_connected_layer'))
        G.add(BatchNormalization(momentum=0.9, name='mini_batch_1'))
        G.add(LeakyReLU(alpha=0.01))
        G.add(Dropout(self.conf.dropout))

        G.add(Reshape(self.conf.coarse_dim, name='low_resolution_output'))
        assert all(G.output_shape[1:] == self.conf.coarse_dim)

        # second block
        G.add(Conv2DTranspose(int(self.conf.coarse_dim[2]/2), 
                              kernel_size=self.conf.filters, 
                              strides=self.conf.stride,
                              kernel_initializer=kinit,
                              padding='same', 
                              use_bias=False, 
                              name='convolve_coarse_input'))
        G.add(BatchNormalization(momentum=0.9, name='mini_batch_2'))
        G.add(LeakyReLU(alpha=0.01))            # alpha: slope coefficient
        G.add(Dropout(self.conf.dropout))

        # third block
        G.add(Conv2DTranspose(int(self.conf.coarse_dim[2]/4), 
                              kernel_size=self.conf.filters, 
                              strides=self.conf.stride,
                              kernel_initializer=kinit, 
                              padding='same', 
                              use_bias=False, 
                              name='convolve_upsampling_1'))
        G.add(BatchNormalization(momentum=0.9, name='mini_batch_3'))
        G.add(LeakyReLU(alpha=0.01))
        G.add(Dropout(self.conf.dropout))

        # outro block
        G.add(Conv2DTranspose(int(self.conf.output_dim[2]), 
                              kernel_size=self.conf.filters, 
                              strides=1, 
                              kernel_initializer=kinit,
                              padding='same', 
                              use_bias=False, 
                              name='generated_output_picture'))
        G.add(Activation('tanh'))
        assert all(G.output_shape[1:] == self.conf.output_dim)

        # compile model
        #G.compile(loss=self.lossG, optimizer=self.optimizer)

        # Save model visualization
        plot_model(G, to_file=self.path_output+'images/generator_visualization.png', show_shapes=True, show_layer_names=True)
        return G

    def Generator_inpaint(self):
        print('Create Generator network...')
        G = models.Sequential()

        kinit = initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        # first block
        G.add(Conv2DTranspose(self.conf.coarse_dim,
                              kernel_size=self.conf.filters, 
                              strides=1,
                              input_dim=self.conf.input_dim,
                              kernel_initializer=kinit, 
                              name='fully_connected_layer'))
        G.add(BatchNormalization(momentum=0.9, name='mini_batch_1'))
        G.add(LeakyReLU(alpha=0.01))
        G.add(Dropout(self.conf.dropout))

        # second block
        G.add(Conv2DTranspose(int(self.conf.coarse_dim[2]/2), 
                              kernel_size=self.conf.filters, 
                              strides=1,
                              kernel_initializer=kinit,
                              padding='same', 
                              use_bias=False, 
                              name='convolve_coarse_input'))
        G.add(BatchNormalization(momentum=0.9, name='mini_batch_2'))
        G.add(LeakyReLU(alpha=0.01))            # alpha: slope coefficient
        G.add(Dropout(self.conf.dropout))

        # third block
        G.add(Conv2DTranspose(int(self.conf.coarse_dim[2]/4), 
                              kernel_size=self.conf.filters, 
                              strides=1,
                              kernel_initializer=kinit, 
                              padding='same', 
                              use_bias=False, 
                              name='convolve_upsampling_1'))
        G.add(BatchNormalization(momentum=0.9, name='mini_batch_3'))
        G.add(LeakyReLU(alpha=0.01))
        G.add(Dropout(self.conf.dropout))

        # outro block
        G.add(Conv2DTranspose(int(self.conf.output_dim[2]), 
                              kernel_size=self.conf.filters, 
                              strides=1, 
                              kernel_initializer=kinit,
                              padding='same', 
                              use_bias=False, 
                              name='generated_output_picture'))
        G.add(Activation('tanh'))
        assert all(G.output_shape[1:] == self.conf.output_dim)

        # compile model
        #G.compile(loss=self.lossG, optimizer=self.optimizer)

        # Save model visualization
        plot_model(G, to_file=self.path_output+'images/generator_visualization.png', show_shapes=True, show_layer_names=True)
        return G


    def Adversary(self):
        print('Create Adversary network...')
        A = models.Sequential()

        kinit = initializers.RandomNormal(mean=0.0, stddev=0.02)

        # first downsample 
        A.add(Conv2D(int(self.conf.coarse_dim[2]/4), 
                     input_shape=self.conf.output_dim, 
                     kernel_size=self.conf.filters, 
                     strides=self.conf.stride,
                     kernel_initializer=kinit,
                     padding='same'))
        A.add(LeakyReLU(alpha=0.01))
        A.add(Dropout(self.conf.dropout))

        # second downsample
        A.add(Conv2D(int(self.conf.coarse_dim[2]/2),
                     kernel_size=self.conf.filters, 
                     strides=self.conf.stride,
                     kernel_initializer=kinit, 
                     padding='same'))
        A.add(LeakyReLU(alpha=0.01))
        A.add(Dropout(self.conf.dropout))
        
        # tird downsample
        A.add(Conv2D(int(self.conf.coarse_dim[2]),
                     kernel_size=self.conf.filters, 
                     strides=1,
                     kernel_initializer=kinit, 
                     padding='same'))
        A.add(LeakyReLU(alpha=0.01))
        A.add(Dropout(self.conf.dropout))

        # classifier
        A.add(Flatten())
        A.add(Dense(self.conf.output_dim[2], activation='sigmoid'))
        
        # compile model
        A.compile(loss=self.lossA, optimizer=self.optimizer)

        # Save model visualization
        plot_model(A, to_file=self.path_output+'images/adversary_visualization.png', show_shapes=True, show_layer_names=True)
        return A