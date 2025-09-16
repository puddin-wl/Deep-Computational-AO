# -*- coding: utf-8 -*-
"""
Created on 2020/09/14

@author: yloffice
"""

## import communal module
import os
import sys
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
## import private module
import dataloader_wf as dataloader
from loss_mse_ssim import loss_mse_ssim

#from network.tools import mse_ssim_loss,mse_ssim_loss1,ssim_loss,ssim_loss_ver2
## For parsing commandline arguments
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.name_model = "RCAN"
args.dataset = r"E:\DL_SR\data\win"
args.num_subframe = 1
args.patch_size = [256, 256]
args.train_batch_size = 1
args.id_epoch = [0, 200]
args.init_learning_rate = 1e-4
args.continue_train = False
args.trained_model = None  #r"Z:\hwl\DL_SR_HY\Checkpoints_win_win_win\model_testRCAN_0026.pkl"
args.SummaryWriter = None
args.cpu_mode = False
model_name = 'model_test'

# Initialize
path_network = './network'
sys.path.append(path_network)

if args.cpu_mode:
    device = torch.device("cpu")
    mode_cpu = True
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mode_cpu = True if device.type == 'cpu' else False
    torch.backends.cudnn.benchmark = True

## Create Checkpoints_win_win_win directory
if not os.path.exists("Checkpoints_win_win_win"):
    os.mkdir("Checkpoints_win_win_win")


## Initialize CNNs
class netPara:
    def __init__(self):
        self.n_colors = 1
        self.n_resgroups = 10
        self.n_resblocks = 3
        self.n_feats = 64
        self.reduction = 16
        self.BMAX = 65535
        self.scale = [1]
        self.rgb_range = 255
        self.res_scale = 1


paraModel = netPara()
NAME_model = args.name_model
#from network.model_fft_2X_ver3_wf import EFDSIM
from network.rcan import RCAN

#from network.model_2to1_mutip_ver5 import mse_ssim_loss
#MSE_SSIM=mse_ssim_loss()
if mode_cpu:
    model = RCAN(paraModel)
else:
    model = RCAN(paraModel).cuda(device)

learning_rate = args.init_learning_rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
SW_continue = args.continue_train
if SW_continue:
    PATH_MODEL = args.trained_model
    model.load_state_dict(torch.load(PATH_MODEL))

## Dataloader
PATH_DATASET = args.dataset
SIZE_PATCH = args.patch_size
SIZE_BATCH = args.train_batch_size
trainset = dataloader.DLSIM_v0(root=PATH_DATASET + '/train_data',
                               num_sub=1,
                               randomCropSize=SIZE_PATCH, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=SIZE_BATCH, shuffle=True)
print("{}:\n{}".format("trainset", trainset))

validationset = dataloader.DLSIM_v0(root=PATH_DATASET + '/test_data',
                                    num_sub=1,
                                    randomCropSize=SIZE_PATCH, train=False)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=SIZE_BATCH, shuffle=False)
print("{}:\n{}".format("validationset", validationset))

PATH_SummaryWriter = args.SummaryWriter
txt_save = "checkpoints_" + model_name + "_1.txt"
if SW_continue:
    writer = SummaryWriter(PATH_SummaryWriter)
else:
    writer = SummaryWriter()
    ftxt = open(os.path.join("./Checkpoints_win_win_win", txt_save), 'w')
    ftxt.write("epoch\t\t" + \
               "mmes_tr\t\t" + "std_tr\t\t" + "loss_tr\t\t" + \
               "mmes_ts\t\t" + "std_ts\t\t" + "loss_ts\n")
    ftxt.close()


# Sub-function
def get_learning_rate(epoch, lim, init_learning_rate=learning_rate):
    lr = init_learning_rate / (2 ** (epoch // lim))
    return lr


# Train
ID_epoch = args.id_epoch
print("<<<Train Begin>>>\n")
for epoch in range(ID_epoch[0], ID_epoch[1]):
    # model train
    model.train()

    lr = get_learning_rate(epoch, 100)
    for p in optimizer.param_groups:
        p['lr'] = lr

    mse_train = []
    loss_train = 0
    for batch_index, (real1x, wf1x) in enumerate(trainloader, 0):
        # cuda mode
        if not mode_cpu:
            srcImg = real1x.cuda(device)
            #wf1x=5*wf1x
            wf = wf1x.cuda(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        #print(wf.shape)
        # forward + backward + optimize
        predImg = model.forward(wf)
        #print(predImg.shape)
        # loss_running = 1000 * (predImg - srcImg).abs().mean() + 1000 * ((predImg - srcImg) ** 2).mean()
        #loss_running=(predImg - srcImg).abs().mean() + 5 * ((predImg-srcImg)**2).mean()+ssim_loss_ver2(srcImg,predImg,wf)
        #loss_running=mse_ssim_loss1(srcImg,predImg,wf)
        #loss_running=5 * ((predImg-srcImg)**2).mean()+ssim_loss_ver2(srcImg,predImg,wf)
        loss_running = loss_mse_ssim(predImg, srcImg)
        loss_running.backward()
        optimizer.step()
        print("[Epoch %04d] [Batch %04d/%04d] [Loss_running: %.3e]" % \
              (epoch, batch_index, len(trainloader), loss_running.item()))
        loss_train += loss_running.item()
        # MSE
        mse_train.append(F.mse_loss(predImg, srcImg).item())

    # model eval
    model.eval()
    mse_test = []
    loss_test = 0
    with torch.no_grad():
        for batch_index, (real1x, wf1x) in enumerate(validationloader, 0):
            # cuda mode
            if not mode_cpu:
                srcImg = real1x.cuda(device)
                wf = wf1x.cuda(device)
            # forward
            predImg = model.forward(wf)
            loss_ = ((predImg - srcImg).abs().mean() + 1 * ((predImg - srcImg) ** 2).mean()).item()
            # loss_=((predImg - srcImg).abs().mean() + 5 * ((predImg-srcImg)**2).mean()+ssim_loss_ver2(srcImg,predImg,wf)).item()
            #loss_=mse_ssim_loss1(srcImg,predImg,wf)
            #loss_=5 * ((predImg-srcImg)**2).mean()+ssim_loss_ver2(srcImg,predImg,wf)
            loss_test += loss_
            # MSE
            mse_test.append(F.mse_loss(predImg, srcImg).item())

    # save checkpoint
    model_save = model_name + NAME_model + "_" + "{:04d}".format(epoch) + ".pkl"
    if epoch % 1 == 0:
        torch.save(model.state_dict(), os.path.join("./Checkpoints_win_win_win", model_save))
    mmse_train = np.mean(mse_train)
    std_train = np.std(mse_train)
    mmse_test = np.mean(mse_test)
    std_test = np.std(mse_test)
    loss_train = loss_train / len(trainloader)
    loss_test = loss_test / len(validationloader)
    ftxt = open(os.path.join("./Checkpoints_win_win_win", txt_save), 'a')
    ftxt.write("{:04d}\t\t".format(epoch) + \
               "{:.3e}\t\t".format(mmse_train) + \
               "{:.3e}\t\t".format(std_train) + \
               "{:.3f}\t\t".format(loss_train) + \
               "{:.3e}\t\t".format(mmse_test) + \
               "{:.3e}\t\t".format(std_test) + \
               "{:.3f}\n".format(loss_test))
    ftxt.close()

    # summary writer
    writer.add_scalar('Loss/Train', loss_train, epoch)
    writer.add_scalar('MMSE/Train', mmse_train, epoch)
    writer.add_scalar('STD/Train', std_train, epoch)
    writer.add_scalar('Loss/Test', loss_test, epoch)
    writer.add_scalar('MMSE/Test', mmse_test, epoch)
    writer.add_scalar('STD/Test', std_test, epoch)
    writer.flush()

# train end
print("<<<Train End>>>\n")
writer.close()
