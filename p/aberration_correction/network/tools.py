import os
from tkinter import X
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import scipy
import math
from pytorch_ssim import ssim
from skimage.metrics import structural_similarity as compare_ssim
import torch
import torch.nn.functional as F
from math import exp
import numpy as np
def fft2d(input, gamma=0.1):
   # temp = torch.permute_dimens(input, (0, 3, 1, 2))
   # temp=input
    #fft = .fft2d(tf.complex(temp, tf.zeros_like(temp)))
    fft=torch.fft.fft2(torch.complex(input,torch.zeros_like(input)))
    #absfft = tf.pow(tf.abs(fft)+1e-8, gamma)
    absfft=torch.pow(torch.abs(fft)+1e-8, gamma)
   # output = K.permute_dimensions(absfft, (0, 2, 3, 1))
    return torch.fft.fftshift(absfft)
def global_average_pooling2d(layer_in):
    return torch.mean(layer_in,(2,3),keepdim=True)
def gelu(x):
    cdf = 0.5 * (1.0 + torch.erf(x / torch.sqrt(2.0)))
    return x * cdf


def loss_mse_ssim(y_true, y_pred):
    ssim_para = 1e-1 # 1e-2
    mse_para = 1

    # nomolization
    x = y_true
    y = y_pred
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))
    temp_x=x.detach().cpu().numpy().squeeze()
    temp_y=y.detach().cpu().numpy().squeeze()
    temp=compare_ssim(temp_y,temp_x)
   # print(temp)
    temp_ssim=torch.from_numpy(np.array(temp))
    ssim_loss = ssim_para * (1 - torch.mean(temp_ssim))
    mse_loss = mse_para * torch.mean(torch.square(y - x))

    return mse_loss + ssim_loss


def loss_mae_mse(y_true, y_pred):
    mae_para = 0.2
    mse_para = 1

    # nomolization
    x = y_true
    y = y_pred
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))

    mae_loss = mae_para * torch.mean(torch.abs(x-y))
    mse_loss = mse_para * torch.mean(torch.square(y - x))

    return mae_loss + mse_loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim_(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret
def mse_ssim_loss(row,pred,wf):
    ssim_para = 1e-2 # 1e-2
    mse_para = 5
        # nomolization
    x = row
    y = pred
    wf=wf
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))
    # print(temp)
    temp_ssim=ssim_(x,y)
    ssim_loss = ssim_para * (1 - torch.mean(temp_ssim))
    mse_loss = mse_para * torch.mean(torch.square(y - x))
    #print(ssim_loss)
    return mse_loss + ssim_loss
def mse_ssim_loss1(row,pred,wf):
    ssim_para = 5e-3 # 1e-2
    mse_para = 5
        # nomolization
    x = row
    y = pred
    wf=wf
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))
    wf= (wf - torch.min(wf)) / (torch.max(wf) - torch.min(wf))
    # print(temp)
    temp_ssim1=(1-torch.mean(ssim_(x,y)))
    temp_ssim2=(1-torch.mean(ssim_(x,wf)))
    #ssim_loss = ssim_para * (1 - torch.mean(temp_ssim))
    ssim_loss = ssim_para * (temp_ssim1/temp_ssim2)
    mse_loss = mse_para * torch.mean(torch.square(y - x))
    #print(ssim_loss)
    return mse_loss + ssim_loss
def ssim_loss(row,pred,wf):
    ssim_para=1e-2
    x = row
    y = pred
    wf=wf
    temp_ssim1=(1-torch.mean(ssim_(x,y)))
    temp_ssim2=(1-torch.mean(ssim_(x,wf)))
    ssim_loss = ssim_para * (temp_ssim1/temp_ssim2)
    return ssim_loss
def ssim_loss_ver2(row,pred,wf):
    ssim_para=5e-3
    x = row.detach().cpu().numpy().squeeze()
    y = pred.detach().cpu().numpy().squeeze()
    wf=wf.detach().cpu().numpy().squeeze()
    temp_ssim1=(1-compare_ssim(x,y))
    temp_ssim2=(1-compare_ssim(x,wf))
    ssim_loss = ssim_para * (temp_ssim1/temp_ssim2)
    ssim_out=torch.from_numpy(np.array(ssim_loss))
    #print(ssim_out)
    return ssim_out