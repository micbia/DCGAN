import sys
from dcganNet import DCGANnetwork

script, init_file = sys.argv

net = DCGANnetwork(CONFIG_FILE=init_file)
net.DCGAN()         # create network and compile
net.TrainDCGAN()

print('... %s script ended' %script[:-3])