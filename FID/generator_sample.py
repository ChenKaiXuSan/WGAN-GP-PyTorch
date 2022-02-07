# %%
import sys
sys.path.append("..")
from utils.utils import *
from models.wgan_gp import Generator
from dataset.dataset import getdDataset
import torch
import os

# %%
# PATH
CHECK_POINT = "/GANs/WGAN-GP-PyTorch/checkpoint/1217_fashion_wgangp_bs64_matsumoto_0"
BATCH_SIZE = 10000
SAVE_SAMPLE_PATH = "/GANs/WGAN-GP-PyTorch/samples/1217_fashion_wgangp_bs64_matsumoto_0"


# %%
fixed_z = tensor2var(torch.randn(BATCH_SIZE, 100))  # （*, 100）

class opt:
    dataroot = '../../data'
    dataset = 'fashion'
    img_size = 64
    batch_size = BATCH_SIZE
#%%
# get real images
data_loader = getdDataset(opt)
real_images = next(iter(data_loader))[0]
# %%
for i in range(0, 10001, 500):
    # init generator
    generator = Generator(channels=1).cuda()

    EPOCH_NAME = str(i) + '.pth.tar'
    CHECKPOINT_PATH = os.path.join(CHECK_POINT, EPOCH_NAME)
    print("now checkpoint:" + CHECKPOINT_PATH)

    # load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH)

    generator.load_state_dict(checkpoint['G_state_dict'])
    epoch = checkpoint['epoch']
    
    # output fake images
    with torch.no_grad():
        fake_images = generator(fixed_z)

    # real_images = get_real_images(opt)

    save_sample_one_image(SAVE_SAMPLE_PATH, real_images, fake_images, epoch)

    torch.cuda.empty_cache()


# %%
