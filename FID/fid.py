'''
use the pytorch-fid to calculate the FID score.
calculate with 10k images, in different 9900 epochs, separately.
'''
from calc_fid import *
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# %%
PATH = "/GANs/WGAN-GP-PyTorch/samples/"
FILE_NAME = "1217_fashion_wgangp_bs64_matsumoto_0"

fid_one_list(PATH, FILE_NAME)

# fid_all_list("/GANs/WGAN-div-PyTorch-/samples/1216_cifar10_bs64_matsumoto_1")
