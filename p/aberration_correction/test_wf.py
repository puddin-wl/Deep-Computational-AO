# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:45:17 2020

@author: yloffice
"""

import os
import sys
import math
import argparse
import numpy as np
import cv2 as cv
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
#from sklearn import preprocessing
# user's modules
import dataloader_wf as dataloader
from scipy.io import savemat as savemat
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import time

# Initialize
path_network = './network'
out_info_txt = 'test_info.txt'
sys.path.append(path_network)


def psnr1(img1, img2):
    img1 = np.float64(abs(img1))
    img2 = np.float64(abs(img2))
    #img1 = np.uint8(abs(img1))
    #img2 = np.uint8(abs(img2))
    mse = np.mean(np.square(img1 - img2))
    #print(mse)
    a = np.max(np.max(img1))
    b = np.max(np.max(img2))
    if b > a:
        a = b
    PIXEL_MAX = a
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def cal1(row1, range1):
    row1 = row1.detach().cpu().numpy().squeeze()
    range1 = range1.detach().cpu().numpy().squeeze()
    temp1 = row1 * range1
    temp2 = temp1.sum(0) / np.square(range1.sum(0))
    return temp2


## For parsing commandline arguments
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.path_model = r"E:\DL_SR\Checkpoints_win_win_win\model_testRCAN_0050.pkl"
args.dataset = r"E:\DL_SR\data\noise"
args.output = r"E:\DL_SR\data\noise_restored"
args.size_img = [256, 256]
args.num_subframe = 1
args.batch_size = 1
args.cpu_mode = False
args.patch_size = [256, 256]

## Diver mode
if args.cpu_mode:
    device = torch.device("cpu")
    mode_cpu = True
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mode_cpu = True if device.type == 'cpu' else False
    torch.backends.cudnn.benchmark = True

torch.backends.cudnn.benchmark = True

## Create output directory
dir_output = args.output
if not os.path.exists(dir_output):
    os.mkdir(dir_output)


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
#from network.model_fft_2X_ver3_wf import EFDSIM
from network.rcan import RCAN

#from network.model_2to1_mutip_ver5 import mse_ssim_loss
#MSE_SSIM=mse_ssim_loss()
if mode_cpu:
    model = RCAN(paraModel)
else:
    model = RCAN(paraModel).cuda(device)

## load CNNs file
PATH_MODEL = args.path_model
if mode_cpu:
    model.load_state_dict(torch.load(PATH_MODEL, map_location='cpu'))
else:
    model.load_state_dict(torch.load(PATH_MODEL))

## Dataloader
PATH_DATASET = args.dataset
SIZE_BATCH = args.batch_size
TRANSFOM = transforms.Compose([transforms.ToTensor()])
NUM_SUB = args.num_subframe
SIZE_CROP = args.size_img
SIZE_PATCH = args.patch_size
testset = dataloader.DLSIM_v0(root=PATH_DATASET,
                              num_sub=1,
                              randomCropSize=SIZE_PATCH, train=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=SIZE_BATCH, shuffle=False)
print("{}:\n{}\n".format("testset", testset))

mse = []
psnr = []
ssim = []
mse_row_src = []
psnr_row_src = []
ssim_row_src = []
# model eval
print("<<<Test Begin>>>\n")
model.eval()
with torch.no_grad():
    for batch_index, (real2x, wf1x) in enumerate(testloader, 1):
        # forward
        if mode_cpu:
            srcimg = real2x
            wf1x = wf1x
            wf2x = real2x
            predimg = model.forward(wf1x)

        else:
            srcimg = real2x.cuda(device)
            wf1x = wf1x.cuda(device)
            wf2x = real2x.cuda(device)
            predimg = model.forward(wf1x)

        # MSE and PSNR
        # mse_ = F.mse_loss(predimg, srcimg).item()
        #psnr_ = -10*math.log(mse_, 10)
        #psnr_=psnr1(predimg.detach().cpu().numpy().squeeze(),srcimg.cpu().numpy().squeeze())
        #ssim_=calculate_ssim(predimg.detach().cpu().numpy().squeeze(), srcimg.cpu().numpy().squeeze())
        #psnr_=psnr1(srcimg.cpu().numpy().squeeze(),predimg.detach().cpu().numpy().squeeze())
        #ssim_=calculate_ssim(srcimg.cpu().numpy().squeeze(),predimg.detach().cpu().numpy().squeeze())
        #ssim_=compare_ssim(srcimg.cpu().numpy().squeeze(),predimg.detach().cpu().numpy().squeeze())

        #mse_row_src_=F.mse_loss(wf2x, srcimg).item()
        # psnr_row_src_=psnr1(srcimg.cpu().numpy().squeeze(),wf2x.detach().cpu().numpy().squeeze())
        # ssim_row_src_=compare_ssim(srcimg.cpu().numpy().squeeze(),wf2x.detach().cpu().numpy().squeeze())

        pred_show = 100 * predimg.detach().cpu().numpy().squeeze()
        #pred_img=(pred_show-np.max(pred_show))/(np.max(pred_show)-np.min(pred_show))
        pred_img = pred_show / (np.max(np.max(pred_show)))
        src_show = srcimg.cpu().numpy().squeeze()
        row_show = wf1x.detach().cpu().numpy().squeeze()
        # row_show=cal1(rowimg,rowrange)
        src_show = src_show * 65535
        src_show = src_show.astype('uint16')
        pred_img = pred_img * 65535
        pred_img = pred_img.astype('uint16')
        # save pred and src
        name_file = "Index{:03d}".format(batch_index)
        # cv.imwrite(os.path.join(dir_output, name_file + 'src.tiff'), src_show)
        cv.imwrite(os.path.join(dir_output, name_file + 'pred.tiff'), pred_img)
        # cv.imwrite(os.path.join(dir_output, name_file + 'wf.tiff'), row_show)
        f = open(os.path.join(dir_output, out_info_txt), 'a')
