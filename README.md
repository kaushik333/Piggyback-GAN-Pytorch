# Piggyback-GAN-Pytorch

## Introduction
The CycleGAN and Pix2Pix code is mostly taken from ![here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 

The main contribution of this repository is the implementation of PiggybackConv and PiggybackTransposeConv module in ```./models/networks.py```. These are custom convolution modules that have unconstrained filters and piggyback filters. As described in the paper, there are only unconstrained filters for Task 1. In the subsequent tasks, there are both piggyback and unconstrained filters. 

The repository also implements parallelism through nn.DistributedDataParallel. 

## Results visualization 
The task wise filter learning is illustrated below in the fgure taken from the paper:
![](./README_figures/pb_gan_pic.png)

## Instructions to run
First, run the following to setup the environment: 
```
conda env create -f environment.yml
```

Download 4 cycleGAN datasets:
```
bash ./datasets/download_cyclegan_dataset.sh maps
bash ./datasets/download_cyclegan_dataset.sh facades
bash ./datasets/download_cyclegan_dataset.sh vangogh2photo
```
For cityscapes, read instructions on how to download and prepare, from: ```./datasets/prepare_cityscapes_dataset.py```

To perform training, run: 
```
python pb_cycleGAN.py train=True
```

To perform testing from trained model, use:
```
python pb_cycleGAN.py train=False
```

Todo: 
1. Write README
2. Include experiemnts on pix2pix
3. Include argparse to take in options from cmd line.
4. Add dataset download scripts. 
5. Calculate FID and tabulate results.