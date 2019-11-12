import sys
from ganNet import GANnetwork

script, init_file = sys.argv

net = GANnetwork(CONFIG_FILE=init_file)
net.GAN()         # create network and compile
net.TrainGAN()

print('... %s script ended' %script[:-3])
