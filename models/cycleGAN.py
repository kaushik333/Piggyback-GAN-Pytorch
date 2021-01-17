# %%
import torch
import torch.nn as nn
from models import networks 
import itertools
from utils.utils import ImageBuffer
from utils.utils import save_image, tensor2im
# %%
class CycleGAN(nn.Module):
    def __init__(self, opt, device):
        super(CycleGAN, self).__init__()

        self.device = device
        self.opt = opt

        self.netG_A = networks.define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG, self.opt.norm,
                                        self.opt.dropout, self.opt.init_type, self.opt.init_gain, self.opt.task_num, self.opt.netG_A_filter_list)
        self.netG_B = networks.define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG, self.opt.norm,
                                        self.opt.dropout, self.opt.init_type, self.opt.init_gain, self.opt.task_num, self.opt.netG_B_filter_list)

        if opt.train:
            self.netD_A = networks.define_D(self.opt.input_nc, self.opt.ndf, self.opt.netD, self.opt.norm, self.opt.init_type, self.opt.init_gain)
            self.netD_B = networks.define_D(self.opt.input_nc, self.opt.ndf, self.opt.netD, self.opt.norm, self.opt.init_type, self.opt.init_gain)

            self.fake_A_pool = ImageBuffer(self.opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImageBuffer(self.opt.pool_size)  # create image buffer to store previously generated images

            self.criterionGAN = networks.GANLoss(self.opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_train_images(self, epoch):
        save_image(tensor2im(self.real_A), self.opt.img_save_path + f"/real_A_epoch_{epoch}.png")
        save_image(tensor2im(self.real_B), self.opt.img_save_path + f"/real_B_epoch_{epoch}.png")
        save_image(tensor2im(self.rec_A), self.opt.img_save_path + f"/rec_A_epoch_{epoch}.png")
        save_image(tensor2im(self.rec_B), self.opt.img_save_path + f"/rec_B_epoch_{epoch}.png")
        save_image(tensor2im(self.idt_A), self.opt.img_save_path + f"/idt_A_epoch_{epoch}.png")
        save_image(tensor2im(self.idt_B), self.opt.img_save_path + f"/idt_B_epoch_{epoch}.png")

    def save_test_images(self, idx):
        save_image(tensor2im(self.real_A), self.opt.img_save_path + f"/img_{idx:04d}_real_A.png")
        save_image(tensor2im(self.rec_A), self.opt.img_save_path + f"/img_{idx:04d}_rec_A.png")
        save_image(tensor2im(self.fake_B), self.opt.img_save_path + f"/img_{idx:04d}_trans_A2B.png")
        save_image(tensor2im(self.real_B), self.opt.img_save_path + f"/img_{idx:04d}_real_B.png")
        save_image(tensor2im(self.rec_B), self.opt.img_save_path + f"/img_{idx:04d}_rec_B.png")
        save_image(tensor2im(self.fake_A), self.opt.img_save_path + f"/img_{idx:04d}_trans_B2A.png")

    
    def update_learning_rate(self):
        
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake, retain_graph=True):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward(retain_graph=retain_graph)
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, retain_graph=False)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
