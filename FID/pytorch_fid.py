'''
use the pytorch-fid to calculate the FID score.
calculate with 10k images, in different 9900 epochs, separately
'''

from calc_fid import *

# PATH
PATH = "/home/xchen/GANs/WGAN-GP-PyTorch/samples/"
FILE_NAME = "1123_wgan_gp_mnist_hatano"

fid_one_list(PATH, FILE_NAME)
