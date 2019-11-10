import numpy as np
from matplotlib.image import imread
from glob import glob
from PIL import Image

class LoadData:
    def __init__(self, PATH_DATA, PATH_MASK='inputs/mask/testing_mask_dataset/'):
        self.path_mask = PATH_MASK
        self.path_data = PATH_DATA
        self.dataset = self.LoadTrainData()
        
        self.masked_dataset = 0

    def LoadTrainData(self, fmt='jpg'):
        images = np.array(glob(self.path_mask+'*.'+fmt))
        nr_imgs, imgs_shape = images.size, imread(images[0]).shape
        dataset = np.zeros(tuple(np.append(nr_imgs, np.array(imgs_shape))))

        for i, img in enumerate(images):
            dataset[i] = imread(img)
        
        # Rescale images values between -1 and 1 
        dataset = (dataset - 127.5)/127.5
        return dataset  

    def LoadMask(self, fmt='jpg'):
        dataset_shape = tuple(np.shape(self.dataset)[:-1])
        maskset = np.zeros(dataset_shape)

        # Rescale mask hight and size to match the images
        size = tuple(dataset_shape[1:])


        for i, img in enumerate(maskset):
            dataset[i] = imread(img)
        
        # Rescale images values between -1 and 1 
        dataset = (dataset - 127.5)/127.5

        return dataset
        #11999
        
        size = 100, 100 
        im = Image.open("inputs/mask/small_mask_dataset/05303.png") 
        im_resized = im.resize(size, Image.ANTIALIAS) 
        im_resized.save("inputs/mask/small_mask_dataset/test-05305.png", "PNG") 