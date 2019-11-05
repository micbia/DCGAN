import numpy as np, os, configparser, matplotlib.pylab as plt

from time import time
from datetime import datetime
from tqdm import tqdm

from keras.layers import Dense, Dropout, Flatten, Reshape, BatchNormalization, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.datasets import mnist
from keras.utils import multi_gpu_model, plot_model
from keras import models, optimizers, initializers, callbacks, regularizers

from config.net_config import NetworkConfig
from utils.metrics import LossAdversary, LossGenerator

class DCGANnetwork:
    def __init__(self, PATH_CONFIG='./config/', PATH_OUTPUT=None):
        print('DCGAN network')

        # Configure networks
        self.path_config = PATH_CONFIG
        self.conf = NetworkConfig(self.path_config+'gan.ini')
        self.optimizer = optimizers.Adam(lr=self.conf.lr, beta_1=self.conf.beta1)   # see Radford et al. 2015
        self.loss = 'binary_crossentropy'

        # Create output directory and sub-directories
        if PATH_OUTPUT != None:
            self.path_output = PATH_OUTPUT + datetime.now().strftime('%d-%mT%H-%M-%S')
            if not os.path.exists(self.path_output):
                os.makedirs(self.path_output)
                os.makedirs(self.path_output+'/model')
                os.makedirs(self.path_output+'/images')
                os.makedirs(self.path_output+'/checkpoints')
                os.makedirs(self.path_output+'/checkpoints/weights')
        else:
            self.path_output = datetime.now().strftime('%d-%mT%H-%M-%S')
            if not os.path.exists(self.path_output):
                os.makedirs(self.path_output)
                os.makedirs(self.path_output+'/model')
                os.makedirs(self.path_output+'/images')
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
        G.add(BatchNormalization(name='mini_batch_1'))
        G.add(LeakyReLU(alpha=self.conf.alpha))

        G.add(Reshape(self.conf.coarse_dim, name='low_resolution_output'))
        assert all(G.output_shape[1:] == self.conf.coarse_dim)
        print(type(self.conf.coarse_dim))
        # second block
        G.add(Conv2DTranspose(int(self.conf.coarse_dim[2]/2), 
                              kernel_size=self.conf.kernel_size, 
                              strides=1, 
                              padding='same', 
                              use_bias=False, 
                              name='convolve_coarse_input'))
        assert all(G.output_shape[1:] == np.array([self.conf.coarse_dim[0], self.conf.coarse_dim[1], self.conf.coarse_dim[2]/2]))
        G.add(BatchNormalization(name='mini_batch_2'))
        G.add(LeakyReLU(alpha=self.conf.alpha))            # alpha: slope coefficient
        
        # third block
        G.add(Conv2DTranspose(int(self.conf.coarse_dim[2]/4), 
                              kernel_size=self.conf.kernel_size, 
                              strides=self.conf.upsampl_size, 
                              padding='same', 
                              use_bias=False, 
                              name='convolve_upsampling_1'))
        assert all(G.output_shape[1:] == np.array([self.conf.coarse_dim[0]*2, self.conf.coarse_dim[1]*2, self.conf.coarse_dim[2]/4]))
        G.add(BatchNormalization(name='mini_batch_3'))
        G.add(LeakyReLU(alpha=self.conf.alpha))
        
        # outro block
        G.add(Conv2DTranspose(int(self.conf.output_dim[2]), 
                              kernel_size=self.conf.kernel_size, 
                              strides=self.conf.upsampl_size, 
                              activation='tanh',
                              padding='same', 
                              use_bias=False, 
                              name='generated_output_picture'))
        assert all(G.output_shape[1:] == self.conf.output_dim)

        # compile model
        G.compile(loss=self.loss, optimizer=self.optimizer)

        # Save model visualization
        plot_model(G, to_file=self.path_output+'model/generator_visualization.png', show_shapes=True, show_layer_names=True)

        return G

    def Adversary(self):
        print('Create Adversary network...')

        A = models.Sequential()

        # first block
        A.add(Conv2D(int(self.conf.coarse_dim[2]/4), 
                     input_shape=self.conf.output_dim, 
                     kernel_size=self.conf.kernel_size, 
                     strides=self.conf.upsampl_size,
                     kernel_initializer=initializers.RandomNormal(stddev=0.02),
                     padding='same'))
        A.add(LeakyReLU(alpha=self.conf.alpha))
        A.add(Dropout(self.conf.dropout))

        # second block
        A.add(Conv2D(int(self.conf.coarse_dim[2]/2),
                     kernel_size=self.conf.kernel_size, 
                     strides=self.conf.upsampl_size, 
                     padding='same'))
        A.add(LeakyReLU(alpha=self.conf.alpha))
        A.add(Dropout(self.conf.dropout))

        # outro block
        A.add(Flatten())
        A.add(Dense(self.conf.output_dim[2], activation='sigmoid'))
        
        # compile model
        A.compile(loss=self.loss, optimizer=self.optimizer)
        A.trainable = False

        # Save model visualization
        plot_model(A, to_file=self.path_output+'model/adversary_visualization.png', show_shapes=True, show_layer_names=True)

        return A

    def DCGAN(self):
        self.generator = self.Generator()
        self.adversary = self.Adversary()
        
        if(self.conf.resume_path != None and self.conf.resume_epoch != 0):
            # load checkpoint weights 
            self.generator.load_weights('%sweights/model_weights-ep%d.h5' %(self.conf.resume_path, self.conf.resume_epoch))
            self.adversary.load_weights('%sweights/model_weights-ep%d.h5' %(self.conf.resume_path, self.conf.resume_epoch))
            print('Adversary and Generator Model weights resumed from:\tmodel_weights-ep%d.h5' %(self.conf.resume_epoch))
            
            # copy logs checkpoints
            os.system('cp %s*ep-%d.txt %scheckpoints/' %(self.conf.resume_path, self.conf.resume_epoch, self.path_output))
        else:
            print('GAN Model Created')

        # define input shape
        inputGAN = Input(shape=(self.conf.input_dim,))
        
        # Create GAN network by merging the generator and adversary network
        self.gan = models.Model(inputs=inputGAN, outputs=self.adversary(self.generator(inputGAN)))
        
        # compile network
        self.gan.compile(loss=self.loss, optimizer=self.optimizer)

        # Save model visualization
        plot_model(self.gan, to_file=self.path_output+'model/gan_visualization.png', show_shapes=True, show_layer_names=True)
        
        return self.gan

    def CreateCheckpoint(self, epch, prev_epch):
        self.generator.save_weights('%scheckpoints/weights/generator_weights_ep-%d.h5' %(self.path_output, epch))
        self.adversary.save_weights('%scheckpoints/weights/adversary_weights_ep-%d.h5' %(self.path_output, epch))

        # delete previous losses checkpoint
        os.remove('%scheckpoints/lossG_ep-%d.txt' %(self.path_output, prev_epch))
        os.remove('%scheckpoints/lossA_ep-%d.txt' %(self.path_output, prev_epch))
        os.remove('%scheckpoints/lossAr_ep-%d.txt' %(self.path_output, prev_epch))
        os.remove('%scheckpoints/lossAf_ep-%d.txt' %(self.path_output, prev_epch))

        # save new losses checkpoint
        np.savetxt('%scheckpoints/lossG_ep-%d.txt' %(self.path_output, epch), self.loss_G)
        np.savetxt('%scheckpoints/lossA_ep-%d.txt' %(self.path_output, epch), self.loss_A)
        np.savetxt('%scheckpoints/lossAr_ep-%d.txt' %(self.path_output, epch), self.loss_A_real)
        np.savetxt('%scheckpoints/lossAf_ep-%d.txt' %(self.path_output, epch), self.loss_A_fake)

    def PlotLoss(self, epch):
        plt.figure(figsize=(10, 8))
        plt.plot(self.loss_A, label='tot Adversary loss', c='darkred', )
        plt.plot(self.loss_A_real, label='real Adversary loss', c='tomato')
        plt.plot(self.loss_A_fake, label='fake Adversary loss', c='lightcoral')
        plt.plot(self.loss_G, label='Generator loss', c='darkgreen')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('%simages/dcgan_loss_ep-%d.png' %(self.path_output, epch))

    def TrainDCGAN(self):
        
        if(self.conf.dataset == 'mnist'):
            # Load MNIST data
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            # Rescale -1 to 1
            x_train = (x_train.astype(np.float32) - 127.5)/127.5
            x_train = x_train[:, :, :, np.newaxis]      # shape : (trainsize, 28, 28, 1)
        else:
            print('to think about it...')    

        self.loss_G = []
        self.loss_A = []
        self.loss_A_real = []
        self.loss_A_fake = []
        prev_epoch = 0

        for ep in range(self.conf.epochs):
            print('--- Epoch %d ---' %ep)
            for bt in tqdm(range(self.conf.batch_size)):
                # create random array of images (same size of batch) to train network on
                real_images = x_train[np.random.randint(0, x_train.shape[0], size=self.conf.batch_size)]
                
                # create random input noise in order to generate fake image
                noise = np.random.normal(loc=0., scale=1., size=[self.conf.batch_size, self.conf.input_dim])
                fake_images = self.generator.predict(noise)

                # smooth label as additional noise, it has been proven that train better network (Salimans et al. 2016)
                fake_label = np.random.uniform(low=0., high=0.3, size=self.conf.batch_size)
                real_label = np.random.uniform(low=0.7, high=1.2, size=self.conf.batch_size)
                
                # train adversary network, with separated mini-batchs, see Ioffe et al. 2015
                self.adversary.trainable = True
                loss_real = self.adversary.train_on_batch(real_images, real_label)
                loss_fake = self.adversary.train_on_batch(fake_images, fake_label)
                self.loss_A.append(np.mean([loss_real, loss_fake])), self.loss_A_real.append(loss_real), self.loss_A_fake.append(loss_fake)
                self.adversary.trainable = False

                # train generator network
                noise = np.random.normal(loc=0., scale=1., size=[self.conf.batch_size, self.conf.input_dim])
                real_label = np.random.uniform(low=0.7, high=1.2, size=self.conf.batch_size)
                loss_gen = self.gan.train_on_batch(noise, real_label)
                self.loss_G.append(loss_gen)

            print('Adversary:\t tot_loss = %.3f\n\t\treal_loss = %.3f\n\t\tfake_loss = %.3f' %(self.loss_A[ep], self.loss_A_real[ep], self.loss_A_fake[ep]))
            print('Generator:\t tot_loss = %.3f' %self.loss_G[ep])
            
            ''' 
            if(ep%5 == 0 and ep != 0):
                self.CreateCheckpoint(epch=ep, prev_epch=prev_epoch)
                self.PlotLoss(epch=ep)
                prev_epoch = ep
            '''

        # save final losses
        np.savetxt('%slossG.txt' %self.path_output, self.loss_G)
        np.savetxt('%slossA.txt' %self.path_output, self.loss_A)
        np.savetxt('%slossAr.txt' %self.path_output, self.loss_A_real)
        np.savetxt('%slossAf.txt' %self.path_output, self.loss_A_fake)
        return 0