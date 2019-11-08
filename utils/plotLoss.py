import numpy as np, matplotlib.pyplot as plt, os 
from sys import argv

script, path = argv

os.chdir(path)

# Load Data
loss_A = np.loadtxt('lossA.txt')
loss_A_real = np.loadtxt('lossAr.txt')
loss_A_fake = np.loadtxt('lossAf.txt')
loss_G = np.loadtxt('lossG.txt') 

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
plt.savefig('images/dcgan_loss.png', bbox_inches='tight')
