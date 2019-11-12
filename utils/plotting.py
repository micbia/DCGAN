import numpy as np, matplotlib.pyplot as plt, os, sys 
from glob import glob
from moviepy.editor import ImageSequenceClip

def PlotLosses(path_output):
    loss_A = np.loadtxt(path_output+'lossA.txt')
    loss_A_real = np.loadtxt(path_output+'lossAr.txt')
    loss_A_fake = np.loadtxt(path_output+'lossAf.txt')
    loss_G = np.loadtxt(path_output+'lossG.txt') 

    # Plot losses
    plt.figure(figsize=(10, 8))
    plt.fill_between(np.array(range(len(loss_A_real))), loss_A_real, loss_A_fake, color='gray', alpha=0.1)
    #plt.plot(self.loss_A, c='tab:red', label='Adversary loss - Total')
    plt.plot(loss_A_real, c='tab:blue', label='Adversary loss - Real')
    plt.plot(loss_A_fake, c='tab:orange' ,label='Adversary loss - Fake')
    plt.plot(loss_G, c='tab:green', label='Generator loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path_output+'images/dcgan_loss.png', bbox_inches='tight')
    return 0


def CreateGif(filename, array, fps=5, scale=1., fmt='gif'):
    ''' Create and save a gif or video from array of images.
        Parameters:
            * filename (string): name of the saved video
            * array (list or string): array of images name already in order, if string it supposed to be the first part of the images name (before iteration integer)
            * fps = 5 (integer): frame per seconds (limit human eye ~ 15)
            * scale = 1. (float): ratio factor to scale image hight and width
            * fmt (string): file extention of the gif/video (e.g: 'gif', 'mp4' or 'avi')
        Return:
            * moviepy clip object
    '''
    if(isinstance(array, str)):
        array = sorted(glob(array+'*.png'), key=os.path.getmtime)
    else:
        pass
    filename += '.'+fmt
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    if(fmt == 'gif'):
        clip.write_gif(filename, fps=fps)
    elif(fmt == 'mp4'):
        clip.write_videofile(filename, fps=fps, codec='mpeg4')
    elif(fmt == 'avi'):
        clip.write_videofile(filename, fps=fps, codec='png')
    else:
        print('Error! Wrong File extension.')
        sys.exit()
    command = os.popen('du -sh %s' % filename)
    print(command.read())
    return clip
