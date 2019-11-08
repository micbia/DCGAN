from PIL import Image
from glob import glob
import os

def ConvertImagesFormat(dir_path, old_format, new_format='RGB', backgr_colour='white'):
    os.chdir(dir_path)
    
    if(backgr_colour == 'white'):
        colour = (255, 255, 255)
    elif(backgr_colour == 'black')
        colour = (0, 0, 0)


    if(old_format == 'RGBA'):
        images = glob('*.png')
    else:
        images = glob('*.'+old_format)


    if(old_format == 'RGBA'):
        for filename in images:
            img = Image.open(filename)
            img.load()
            backgr = Image.new('RGB', img.size, colour)
            backgr.paste(img, mask=img.split()[3])
            backgr.save('%s.jpg' %filename[:-4], 'JPEG', quality=100)
    else:
        img = Image.open(filename)
        img.load()
        img.save('%s.jpg' %filename[:-4], 'JPEG', quality=100)

