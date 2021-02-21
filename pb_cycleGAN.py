# %%
import torch
import torch.nn as nn
import numpy as np
from configs.cycle_GAN_config import CycleGANOptions
import torch.multiprocessing as mp
import torch.distributed as dist
from dataloaders.dataloader import UnalignedDataset
import torch.utils
import os
from models.cycleGAN import CycleGAN
import time
from models.networks import PiggybackConv, PiggybackTransposeConv, load_pb_conv
import copy 
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

# %%
def train(gpu, opt):

    device = torch.device('cuda:{}'.format(gpu)) if gpu>=0 else torch.device('cpu')
    if gpu >= 0:
        torch.cuda.set_device(gpu)
        rank = opt.nr * len(opt.gpu_ids) + gpu	                          
        dist.init_process_group(                                   
            backend='nccl',                                         
            init_method='env://',                                   
            world_size=opt.world_size,                              
            rank=rank                                               
        )           
        model = CycleGAN(opt, device)
        model = model.to(device) 
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        train_dataset = UnalignedDataset(opt)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=rank
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            shuffle=False,            
            num_workers=0,
            pin_memory=True,
            sampler=train_sampler)       

    for epoch in range(opt.start_epoch, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        print("Length of loader is ",len(train_loader))
        for i, data in enumerate(train_loader):
            model.module.set_input(data)
            model.module.optimize_parameters() 
            if (i+1) % 50 == 0 and gpu<=0:
                print_str = (
                    f"Task: {opt.task_num} | Epoch: {epoch} | Iter: {i+1} | G_A: {model.module.loss_G_A:.5f} | "
                    f"G_B: {model.module.loss_G_B:.5f} | cycle_A: {model.module.loss_cycle_A:.5f} | "
                    f"cycle_B: {model.module.loss_cycle_B:.5f} | idt_A: {model.module.loss_idt_A:.5f} | "
                    f"idt_B: {model.module.loss_idt_B:.5f} | D_A: {model.module.loss_D_A:.5f} | "
                    f"D_B: {model.module.loss_D_A:.5f}" 
                )

                print(print_str)
                

        model.module.update_learning_rate()  

        if gpu<=0:
            model.module.save_train_images(epoch)
            save_dict = {'model': model.state_dict(),
                        'epoch': epoch 
                    }
            torch.save(save_dict, opt.ckpt_save_path+'/latest_checkpoint.pt')
        dist.barrier()

    if gpu <= 0:

        netG_A_layer_list = list(model.module.netG_A)
        netG_B_layer_list = list(model.module.netG_B)

        conv_A_idx = 0
        for layer in netG_A_layer_list:
            if isinstance(layer, PiggybackConv) or isinstance(layer, PiggybackTransposeConv):
                layer.unc_filt.requires_grad = False
                if opt.task_num == 1:
                    opt.netG_A_filter_list.append([layer.unc_filt.detach().cpu()])
                elif opt.task_num == 2:
                    opt.netG_A_filter_list[conv_A_idx].append(layer.unc_filt.detach().cpu())
                    opt.weights_A.append([layer.weights_mat.detach().cpu()])
                    conv_A_idx += 1
                else:
                    opt.netG_A_filter_list[conv_A_idx].append(layer.unc_filt.detach().cpu())
                    opt.weights_A[conv_A_idx].append(layer.weights_mat.detach().cpu())
                    conv_A_idx += 1
                    
                
        conv_B_idx = 0
        for layer in netG_B_layer_list:
            if isinstance(layer, PiggybackConv) or isinstance(layer, PiggybackTransposeConv):
                layer.unc_filt.requires_grad = False
                if opt.task_num == 1:
                    opt.netG_B_filter_list.append([layer.unc_filt.detach().cpu()])
                elif opt.task_num == 2:
                    opt.netG_B_filter_list[conv_B_idx].append(layer.unc_filt.detach().cpu())
                    opt.weights_B.append([layer.weights_mat.detach().cpu()])
                    conv_B_idx += 1
                else:
                    opt.netG_B_filter_list[conv_B_idx].append(layer.unc_filt.detach().cpu())
                    opt.weights_B[conv_B_idx].append(layer.weights_mat.detach().cpu())
                    conv_B_idx += 1

        savedict_task = {'netG_A_filter_list':opt.netG_A_filter_list, 
                            'netG_B_filter_list':opt.netG_B_filter_list,
                            'weights_A':opt.weights_A,
                            'weights_B':opt.weights_B
                        }

        torch.save(savedict_task, opt.ckpt_save_path+'/filters.pt')

        del netG_A_layer_list
        del netG_B_layer_list
        del opt.netG_A_filter_list
        del opt.netG_B_filter_list
        del opt.weights_A
        del opt.weights_B
    
    dist.barrier()
    del model

    dist.destroy_process_group()

# %%
        
def test(opt, task_idx):

    opt.train = False
    device = torch.device('cpu')
    model = CycleGAN(opt, device)
    model = model.to(device) 
    model.eval()
    test_dataset = UnalignedDataset(opt)
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,            
            num_workers=4,
            pin_memory=True)

    model.netG_A = load_pb_conv(model.netG_A, opt.netG_A_filter_list, opt.weights_A, task_idx)
    model.netG_B = load_pb_conv(model.netG_B, opt.netG_B_filter_list, opt.weights_B, task_idx)

    for i, data in enumerate(test_loader):
        model.set_input(data)   
        model.forward()
        model.save_test_images(i)
        print(f"Task {opt.task_num} : Image {i}")
        if i > 50:
            break

    del model
      

# %%

# if __name__ == '__main__':

@hydra.main(config_path="configs", config_name="config")
def main(opt : DictConfig):
    
    os.chdir(hydra.utils.get_original_cwd()) 

    # opt = CycleGANOptions()
    tasks = ['./datasets/cityscapes', './datasets/maps', './datasets/facades', './datasets/vangogh2photo']
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    OmegaConf.set_struct(opt, False)

    if opt.train:
        
        start_task = 0
        end_task = len(tasks)

        opt.world_size = len(opt.gpu_ids) * opt.nodes                
        os.environ['MASTER_ADDR'] = 'localhost'              
        os.environ['MASTER_PORT'] = '8888'  

        for task_idx in range(start_task, end_task): 
            
            # Create Task folder 

            opt.task_folder_name = "Task_"+str(task_idx+1)+"_"+tasks[task_idx][41:]+"_"+"cycleGAN"
            opt.image_folder_name = "Intermediate_train_images"
            if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name)):
                os.makedirs(os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name))

            opt.ckpt_save_path = os.path.join(opt.checkpoints_dir, opt.task_folder_name)
            opt.img_save_path = os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name)

            if task_idx == 0:
                netG_A_filter_list = []
                netG_B_filter_list = []
                weights_A = []
                weights_B = []
            else:
                old_task_folder_name = "Task_"+str(task_idx)+"_"+tasks[task_idx-1][41:]+"_"+"cycleGAN"
                print("Loading ", os.path.join(opt.checkpoints_dir, old_task_folder_name)+'/filters.pt')
                filters = torch.load(os.path.join(opt.checkpoints_dir, old_task_folder_name)+'/filters.pt')
                netG_A_filter_list = filters["netG_A_filter_list"]
                netG_B_filter_list = filters["netG_B_filter_list"]
                weights_A = filters["weights_A"]
                weights_B = filters["weights_B"]

            opt.netG_A_filter_list = netG_A_filter_list
            opt.netG_B_filter_list = netG_B_filter_list
            opt.weights_A = weights_A
            opt.weights_B = weights_B

            opt.dataroot = tasks[task_idx]
            opt.task_num = task_idx+1        

            mp.spawn(train, nprocs=len(opt.gpu_ids), args=(opt,))            

    else:
        '''
        We will load the unconstrained filters and the weights ONLY from the last task. 
        This is because, after every task we store the unconstrined filter and weight 
        matrix of that task and all the previous ones. So we will only load from the last one
        which will contain everything we need. 
        '''
        print("In Testing mode")
        start_task = 0
        end_task = len(tasks)
        load_filter_path = opt.checkpoints_dir+f"/Task_{len(tasks)}_{tasks[-1][41:]}_cycleGAN/filters.pt"
        opt.load_filter_path = load_filter_path

        filters = torch.load(opt.load_filter_path)
        opt.netG_A_filter_list = filters["netG_A_filter_list"]
        opt.netG_B_filter_list = filters["netG_B_filter_list"]
        opt.weights_A = filters["weights_A"]
        opt.weights_B = filters["weights_B"]
        opt.image_folder_name = "Test_images"

        for task_idx in range(start_task, end_task):
            print(f"Task {task_idx+1}")

            opt.task_folder_name = "Task_"+str(task_idx+1)+"_"+tasks[task_idx][41:]+"_"+"cycleGAN"
            opt.img_save_path = os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name)
            if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name)):
                    os.makedirs(os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name))

            opt.dataroot = tasks[task_idx]
            opt.task_num = task_idx+1
            test(opt, task_idx)

if __name__ == "__main__":
    main()