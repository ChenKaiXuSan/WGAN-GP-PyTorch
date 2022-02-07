# WGAN-GP-PyTorch
Pytorch implementation of WGAN with gradient penalty (WGAN-GP),

## Overview
This repository contains an Pytorch implementation of WGAN, WGAN-GP, WGAN-DIV and original GAN loss function.
With full coments and my code style.

## About WGAN
If you're new to WassersteinGAN, here's an abstract straight from the paper[1]:

We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to other distances between distributions.

## Dataset 
- MNIST
`python3 main.py --dataset mnist --channels 1`
- FashionMNIST
`python3 main.py --dataset fashion --channels 1`
- Cifar10
`python3 main.py --dataset cifar10 --channels 3`

## Implement
``` python
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Generator                                --                        --
├─Sequential: 1-1                        [64, 512, 4, 4]           --
│    └─ConvTranspose2d: 2-1              [64, 512, 4, 4]           819,200
│    └─BatchNorm2d: 2-2                  [64, 512, 4, 4]           1,024
│    └─ReLU: 2-3                         [64, 512, 4, 4]           --
├─Sequential: 1-2                        [64, 256, 8, 8]           --
│    └─ConvTranspose2d: 2-4              [64, 256, 8, 8]           2,097,152
│    └─BatchNorm2d: 2-5                  [64, 256, 8, 8]           512
│    └─ReLU: 2-6                         [64, 256, 8, 8]           --
├─Sequential: 1-3                        [64, 128, 16, 16]         --
│    └─ConvTranspose2d: 2-7              [64, 128, 16, 16]         524,288
│    └─BatchNorm2d: 2-8                  [64, 128, 16, 16]         256
│    └─ReLU: 2-9                         [64, 128, 16, 16]         --
├─Sequential: 1-4                        [64, 64, 32, 32]          --
│    └─ConvTranspose2d: 2-10             [64, 64, 32, 32]          131,072
│    └─BatchNorm2d: 2-11                 [64, 64, 32, 32]          128
│    └─ReLU: 2-12                        [64, 64, 32, 32]          --
├─Sequential: 1-5                        [64, 1, 64, 64]           --
│    └─ConvTranspose2d: 2-13             [64, 1, 64, 64]           1,024
│    └─Tanh: 2-14                        [64, 1, 64, 64]           --
==========================================================================================
Total params: 3,574,656
Trainable params: 3,574,656
Non-trainable params: 0
Total mult-adds (G): 26.88
==========================================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 127.93
Params size (MB): 14.30
Estimated Total Size (MB): 142.25
==========================================================================================

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Discriminator                            --                        --
├─Sequential: 1-1                        [64, 64, 32, 32]          --
│    └─Conv2d: 2-1                       [64, 64, 32, 32]          1,024
│    └─LayerNorm: 2-2                    [64, 64, 32, 32]          131,072
│    └─LeakyReLU: 2-3                    [64, 64, 32, 32]          --
├─Sequential: 1-2                        [64, 128, 16, 16]         --
│    └─Conv2d: 2-4                       [64, 128, 16, 16]         131,072
│    └─LayerNorm: 2-5                    [64, 128, 16, 16]         65,536
│    └─LeakyReLU: 2-6                    [64, 128, 16, 16]         --
├─Sequential: 1-3                        [64, 256, 8, 8]           --
│    └─Conv2d: 2-7                       [64, 256, 8, 8]           524,544
│    └─LayerNorm: 2-8                    [64, 256, 8, 8]           32,768
│    └─LeakyReLU: 2-9                    [64, 256, 8, 8]           --
├─Sequential: 1-4                        [64, 512, 4, 4]           --
│    └─Conv2d: 2-10                      [64, 512, 4, 4]           2,097,664
│    └─LayerNorm: 2-11                   [64, 512, 4, 4]           16,384
│    └─LeakyReLU: 2-12                   [64, 512, 4, 4]           --
├─Sequential: 1-5                        [64, 1, 1, 1]             --
│    └─Conv2d: 2-13                      [64, 1, 1, 1]             8,192
==========================================================================================
Total params: 3,008,256
Trainable params: 3,008,256
Non-trainable params: 0
Total mult-adds (G): 6.53
==========================================================================================
Input size (MB): 1.05
Forward/backward pass size (MB): 125.83
Params size (MB): 12.03
Estimated Total Size (MB): 138.91
==========================================================================================

```
## Usage
- MNSIT  
`python3 main.py --dataset mnist --channels 1 --version [version] --batch_size [] --adv_loss [] >logs/[log_path]`
- FashionMNIST  
`python3 main.py --dataset fashion --channels 1 --version [version] --batch_size [] --adv_loss [] >logs/[log_path]`
- Cifar10  
`python3 main.py --dataset cifar10 --channels 3 -version [version] --batch_size [] --adv_loss [] >logs/[log_path]`

## FID
FID is a measure of similarity between two datasets of images. It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks. FID is calculated by computing the Fréchet distance between two Gaussians fitted to feature representations of the Inception network.

For the FID, I use the pytorch implement of this repository. [FID score for PyTorch](https://github.com/mseitzer/pytorch-fid)

For the 10k epochs training on different dataset, compare with about 10000 samples, I get the FID: 

| dataset | wgan-gp |
| ---- | ---- |
| MNIST | 67.02213911513545(8900epoch) |
| FASHION-MNIST | 32.801197393610494(8000epoch) | 
| CIFAR10 | 58.6455420134115(500epoch) |
 
> :warning: I dont konw if the FID is right or not, because I cant get the lowwer score like the paper or the other people get it. 

## Reference
1. [WGAN](https://arxiv.org/abs/1701.07875)
2. [WGAN-GP](https://arxiv.org/abs/1704.00028)
3. [WGAN-DIV](https://arxiv.org/abs/1712.01026)
4. [DCGAN](https://arxiv.org/abs/1511.06434)
5. [CT-GAN](https://arxiv.org/abs/1803.01541)(todo)