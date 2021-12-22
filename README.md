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
| MNIST | 67.02213911513545(8900epochs) |
| FASHION-MNIST | null | 
| CIFAR10 | null |
 
> :warning: I dont konw if the FID is right or not, because I cant get the lowwer score like the paper or the other people get it. 

## Reference
1. [WGAN](https://arxiv.org/abs/1701.07875)
2. [WGAN-GP](https://arxiv.org/abs/1704.00028)
3. [WGAN-DIV](https://arxiv.org/abs/1712.01026)
4. [DCGAN](https://arxiv.org/abs/1511.06434)
5. [CT-GAN](https://arxiv.org/abs/1803.01541)(todo)