import numpy as np
from matplotlib.image import imread
from glob import glob
from PIL import Image

from utils.other_utils import BatchSample

class LoadData:
    def __init__(self, PATH_DATA, PATH_MASK='inputs/mask/testing_mask_dataset/'):
        self.path_mask = PATH_MASK
        self.path_data = PATH_DATA
        self.dataset = self.LoadTrainData()

    def LoadTrainData(self, fmt='jpg'):
        images = np.array(glob(self.path_data+'*.'+fmt))
        nr_imgs, imgs_shape = images.size, imread(images[0]).shape
        dataset = np.zeros(tuple(np.append(nr_imgs, np.array(imgs_shape))))

        for i, img in enumerate(images):
            dataset[i] = imread(img)

        return dataset  

    def MaskData(self, fmt='jpg'):
        dataset_shape = tuple(np.shape(self.dataset)[:-1])
        maskset = np.zeros(dataset_shape)

        subsample_masks = BatchSample(sample=np.array(glob(self.path_mask+'*.png')), nr_subsample=dataset_shape[0])
        
        # Rescale mask shape to match the images
        for i in range(subsample_masks.size):
            mask = Image.open(subsample_masks[i])
            mask_resized = mask.resize(tuple(dataset_shape[1:]), Image.ANTIALIAS) 
            maskset[i] = np.array(mask_resized)

        maskset = (maskset - 127.5)/127.5
        maskset = maskset[:, :, :, np.newaxis]

        masked = (self.dataset * maskset -127.5)/127.5

        return masked