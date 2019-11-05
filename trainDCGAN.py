from dcganNet import DCGANnetwork

net = DCGANnetwork(PATH_CONFIG='./config/')
net.DCGAN()         # create network and compile
net.TrainDCGAN()

print('...test ended')