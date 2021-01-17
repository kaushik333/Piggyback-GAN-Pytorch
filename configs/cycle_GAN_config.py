import argparse
import os 

class CycleGANOptions():
    def __init__(self):

        # folder paths
        self.checkpoints_dir = "./checkpoints"

        # device settings
        self.train = False
        self.nodes = 1
        self.gpu_ids = [0,1,2,3]
        self.nr = 0 # ranking within nodes

        # model and arch
        self.ngf = 64
        self.ndf = 64
        self.netD = "basic" # [basic | n_layers | pixel]
        self.netG = "resnet_9blocks" # [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
        self.norm = "instance" # [instance | batch | none]
        self.init_type = "normal" # [normal | xavier | kaiming | orthogonal]
        self.init_gain = 0.02 # scaling factor for normal, xavier and orthogonal.
        self.dropout = False

        # train hyperparams
        self.lambda_A = 10.0
        self.lambda_B = 10.0
        self.lambda_identity = 0.5
        self.start_epoch = 1
        self.n_epochs = 100
        self.n_epochs_decay = 100
        self.beta1 = 0.5
        self.lr = 0.0002
        self.lr_policy = "linear"
        self.gan_mode = "lsgan" # [vanilla| lsgan | wgangp]

        # dataset related options
        self.pool_size = 50
        self.direction = "AtoB"
        self.input_nc = 3
        self.output_nc = 3
        self.batch_size = 2
        self.load_size = 286
        self.crop_size = 256
        self.preprocess = "resize_and_crop" # [resize_and_crop | crop | scale_width | scale_width_and_crop | none]
        self.no_flip = False
