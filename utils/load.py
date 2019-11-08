import numpy as np
from matplotlib.image import imread
from glob import glob

def LoadTrainData(path, fmt='jpg'):
    images = np.array(glob(path+'*.'+fmt))

    nr_imgs = images.size
    imgs_shape = imread(images[0]).shape

    dataset = np.zeros((nr_imgs, imgs_shape[0], imgs_shape[1], imgs_shape[2]))

    for i, img in enumerate(images):
        dataset[i] = imread(img)
    
    # Rescale images values between -1 and 1 
    dataset = (dataset - 127.5)/127.5

    return dataset