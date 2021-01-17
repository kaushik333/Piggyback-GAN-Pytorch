# Piggyback-GAN-Pytorch

The CycleGAN and Pix2Pix code is mostly taken from ![here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). PiggybackGAN is implemented by implementing custom PiggybackConv and PiggybackTransposeConv layers and replacing the original Conv2d and Conv2dTranspose of PyTorch with these.

Todo: 
1. Write README
2. Include experiemnts on pix2pix
3. Include argparse to take in options from cmd line.
4. Add dataset download scripts. 
5. Calculate FID and tabulate results.