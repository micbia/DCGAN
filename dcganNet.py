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
from keras.backend import set_image_dim_ordering

from config.net_config import NetworkConfig
from utils.metrics import LossAdversary, LossGenerator

# set images dimension as TensorFlow does (sample, row, columns, channels)
# NOTE: Theano expects 'channels' at the second dimension (index 1)
set_image_dim_ordering('tf')

class DCGANnetwork:
    def __init__(self, CONFIG_FILE='./config/example.ini', PATH_OUTPUT=None):
        print('DCGAN network')

        # Configure networks
        self.config_file = CONFIG_FILE
        self.conf = NetworkConfig(self.config_file)
        self.optimizer = optimizers.Adam(lr=self.conf.lr, beta_1=self.conf.beta1)   # see Radford et al. 2015
        self.loss = 'binary_crossentropy'

        # Create output directory and sub-directories
        if PATH_OUTPUT != None:
            self.path_output = PATH_OUTPUT + datetime.now().strftime('%d-%mT%H-%M-%S')
            if not os.path.exists(self.path_output):
                os.makedirs(self.path_output)
                os.makedirs(self.path_output+'/model')
                os.makedirs(self.path_output+'/images')
                os.makedirs(self.path_output+'/images/generated_test')
                os.makedirs(self.path_output+'/checkpoints')
                os.makedirs(self.path_output+'/checkpoints/weights')
        else:
            self.path_output = datetime.now().strftime('%d-%mT%H-%M-%S')
            if not os.path.exists(self.path_output):
                os.makedirs(self.path_output)
                os.makedirs(self.path_output+'/model')
                os.makedirs(self.path_output+'/images')
                os.makedirs(self.path_output+'/images/generated_test')
                os.makedirs(self.path_output+'/checkpoints')
                os.makedirs(self.path_output+'/checkpoints/weights')
        self.path_output += '/'


    def Generator(self):
        print('Create Generator network...')
        G = models.Sequential()

        # first block
        G.add(Dense(np.prod(self.conf.coarse_dim), 
                    input_dim=self.conf.input_dim,
                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02), 
                    name='fully_connected_layer'))
        G.add(BatchNormalization(momentum=0.9, name='mini_batch_1'))
        G.add(LeakyReLU(alpha=self.conf.alpha))
        G.add(Dropout(self.conf.dropout))

        G.add(Reshape(self.conf.coarse_dim, name='low_resolution_output'))
        assert all(G.output_shape[1:] == self.conf.coarse_dim)

        # second block
        G.add(Conv2DTranspose(int(self.conf.coarse_dim[2]/2), 
                              kernel_size=self.conf.filters, 
                              strides=self.conf.stride, 
                              padding='same', 
                              use_bias=False, 
                              name='convolve_coarse_input'))
        #assert all(G.output_shape[1:] == np.array([self.conf.coarse_dim[0], self.conf.coarse_dim[1], self.conf.coarse_dim[2]/2]))
        G.add(BatchNormalization(momentum=0.9, name='mini_batch_2'))
        G.add(LeakyReLU(alpha=self.conf.alpha))            # alpha: slope coefficient
        G.add(Dropout(self.conf.dropout))

        # third block
        G.add(Conv2DTranspose(int(self.conf.coarse_dim[2]/4), 
                              kernel_size=self.conf.filters, 
                              strides=self.conf.stride, 
                              padding='same', 
                              use_bias=False, 
                              name='convolve_upsampling_1'))
        #assert all(G.output_shape[1:] == np.array([self.conf.coarse_dim[0]*2, self.conf.coarse_dim[1]*2, self.conf.coarse_dim[2]/4]))
        G.add(BatchNormalization(momentum=0.9, name='mini_batch_3'))
        G.add(LeakyReLU(alpha=self.conf.alpha))
        G.add(Dropout(self.conf.dropout))

        # outro block
        G.add(Conv2DTranspose(int(self.conf.output_dim[2]), 
                              kernel_size=self.conf.filters, 
                              strides=1, 
                              activation='tanh',
                              padding='same', 
                              use_bias=False, 
                              name='generated_output_picture'))
        assert all(G.output_shape[1:] == self.conf.output_dim)

        # compile model
        G.compile(loss=self.loss, optimizer=self.optimizer)

        # Save model visualization
        plot_model(G, to_file=self.path_output+'images/generator_visualization.png', show_shapes=True, show_layer_names=True)
        return G


    def Adversary(self):
        print('Create Adversary network...')
        A = models.Sequential()

        # first block
        A.add(Conv2D(int(self.conf.coarse_dim[2]/4), 
                     input_shape=self.conf.output_dim, 
                     kernel_size=self.conf.filters, 
                     strides=self.conf.stride,
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02),
                     padding='same'))
        A.add(LeakyReLU(alpha=self.conf.alpha))
        A.add(Dropout(self.conf.dropout))

        # second block
        A.add(Conv2D(int(self.conf.coarse_dim[2]/2),
                     kernel_size=self.conf.filters, 
                     strides=self.conf.stride, 
                     padding='same'))
        A.add(LeakyReLU(alpha=self.conf.alpha))
        A.add(Dropout(self.conf.dropout))

        # outro block
        A.add(Flatten())
        A.add(Dense(self.conf.output_dim[2], activation='sigmoid'))
        
        # compile model
        A.compile(loss=self.loss, optimizer=self.optimizer)

        # Save model visualization
        plot_model(A, to_file=self.path_output+'images/adversary_visualization.png', show_shapes=True, show_layer_names=True)
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

        # Create GAN network by merging the generator and adversary network
        self.adversary.trainable = False
        inputGAN = Input(shape=(self.conf.input_dim,))
        outputGAN = self.adversary(self.generator(inputGAN))
        self.gan = models.Model(inputs=inputGAN, outputs=outputGAN)
        
        # compile network
        self.gan.compile(loss=self.loss, optimizer=self.optimizer)

        # Save model visualization
        plot_model(self.gan, to_file=self.path_output+'images/gan_visualization.png', show_shapes=True, show_layer_names=True)
        return self.gan


    def CreateCheckpoint(self, epch, prev_epch):
        self.generator.save_weights('%scheckpoints/weights/generator_weights_ep-%d.h5' %(self.path_output, epch))
        self.adversary.save_weights('%scheckpoints/weights/adversary_weights_ep-%d.h5' %(self.path_output, epch))

        # delete previous losses checkpoint
        if(prev_epch != 0):
            os.remove('%scheckpoints/lossG_ep-%d.txt' %(self.path_output, prev_epch))
            os.remove('%scheckpoints/lossA_ep-%d.txt' %(self.path_output, prev_epch))
            os.remove('%scheckpoints/lossAr_ep-%d.txt' %(self.path_output, prev_epch))
            os.remove('%scheckpoints/lossAf_ep-%d.txt' %(self.path_output, prev_epch))

        # save new losses checkpoint
        np.savetxt('%scheckpoints/lossG_ep-%d.txt' %(self.path_output, epch), self.loss_G)
        np.savetxt('%scheckpoints/lossA_ep-%d.txt' %(self.path_output, epch), self.loss_A)
        np.savetxt('%scheckpoints/lossAr_ep-%d.txt' %(self.path_output, epch), self.loss_A_real)
        np.savetxt('%scheckpoints/lossAf_ep-%d.txt' %(self.path_output, epch), self.loss_A_fake)


    def Plots(self, epch, prev_epch, examples=100, dim=(10, 10)):
        # Plot losses 
        plt.figure(figsize=(10, 8))
        plt.fill_between(np.array(range(len(self.loss_A_real))), self.loss_A_real, self.loss_A_fake, color='gray', alpha=0.1)
        #plt.plot(self.loss_A, c='tab:red', label='Adversary loss - Total')
        plt.plot(self.loss_A_real, c='tab:blue', label='Adversary loss - Real')
        plt.plot(self.loss_A_fake, c='tab:orange' ,label='Adversary loss - Fake')
        plt.plot(self.loss_G, c='tab:green', label='Generator loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('%simages/dcgan_loss_ep-%d.png' %(self.path_output, epch), bbox_inches='tight')
        if(prev_epch != 0):
            os.remove('%simages/dcgan_loss_ep-%d.png' %(self.path_output, prev_epch))

        # Plot generated images
        plt.figure(figsize=(12, 12))
        noise = np.random.normal(0, 1, size=[examples, self.conf.input_dim])
        generatedImages = self.generator.predict(noise)
        for i in range(generatedImages.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generatedImages[i, :, :, 0], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('%simages/generated_test/dcgan_generated_image_epoch_%d.png' %(self.path_output, epch), bbox_inches='tight')

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
            print('--- Epoch %d ---' %(ep+1))
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
                self.adversary.trainable = False

                # train generator network
                noise = np.random.normal(loc=0., scale=1., size=[self.conf.batch_size, self.conf.input_dim])
                real_label = np.random.uniform(low=0.7, high=1.2, size=self.conf.batch_size)
                loss_gen = self.gan.train_on_batch(noise, real_label)
            
            # store losses at the end of every batch cycle
            self.loss_A.append(0.5*(loss_real+loss_fake))
            self.loss_A_real.append(loss_real)
            self.loss_A_fake.append(loss_fake)    
            self.loss_G.append(loss_gen)

            # print losses to monitor trainig
            print('Adversary:\tavrg_loss = %.3f\n\t\treal_loss = %.3f\n\t\tfake_loss = %.3f' %(self.loss_A[ep], self.loss_A_real[ep], self.loss_A_fake[ep]))
            print('Generator:\t     loss = %.3f' %self.loss_G[ep])
            
            if(ep%5 == 0 and ep != 0 or (ep+1) == self.conf.epochs):
                #self.CreateCheckpoint(epch=ep, prev_epch=prev_epoch)
                self.Plots(epch=ep, prev_epch=prev_epoch)
                prev_epoch = ep

        # save final losses and weights
        np.savetxt('%slossG.txt' %self.path_output, self.loss_G)
        np.savetxt('%slossA.txt' %self.path_output, self.loss_A)
        np.savetxt('%slossAr.txt' %self.path_output, self.loss_A_real)
        np.savetxt('%slossAf.txt' %self.path_output, self.loss_A_fake)

        self.generator.save_weights('%smodel/generator_weights.h5' %self.path_output)
        self.adversary.save_weights('%smodel/adversary_weights.h5' %self.path_output)

        return 0