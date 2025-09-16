# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 20:52:18 2020

@author: yloffice
"""

import os
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2 as cv
from PIL import Image
import numpy as np

#%% utils
def _opencv_loader(path, cropArea=None, resizeDim=None, flipCode=None, convert=cv.IMREAD_GRAYSCALE):
    """
    """
    img = cv.imread(path, convert)
    # Resize image if specified
    resized_img = cv.resize(resizeDim) if (resizeDim != None) else img
    # Crop image if crop area specified
    cropped_img = resized_img[cropArea[0]:cropArea[2], cropArea[1]:cropArea[3]] \
        if (cropArea != None) else resized_img
    # Flip image if specified
    flipped_img = cv.flip(cropped_img, flipCode) if (flipCode != None) else cropped_img
    
    return flipped_img
    

def _make_dataset(dir):
    """
    Parameters
    ----------
    dir : TYPE
        DESCRIPTION.
    Returns
    -------
    None.
    """
    
    framesPath = []
    # Find and loog over all the clips in root 'dir'.
    for index, folder in enumerate(os.listdir(dir)):
        clipsFolderPath = os.path.join(dir, folder)
        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue
        framesPath.append([])
        # Find and loop over all the frames inside the clip.
        for image in sorted(os.listdir(clipsFolderPath)):
            # Add path to list.
            framesPath[index].append(os.path.join(clipsFolderPath, image))
            
    return framesPath

class DLSIM_v0(data.Dataset):
    """
    A dataloader for loading N samples arranged in this way:
        |-- clip0
            |-- source image
            -----------------
            |-- SIM row frame00
            |-- SIM row frame01
            :
            |-- SIM row frameN-1
        |-- clip1
            |-- source image
            -----------------
            |-- SIM row frame00
            |-- SIM row frame01
            :
            |-- SIM row frameN-1
        :
        :
        |-- clipN
            |-- source image
            -----------------
            |-- SIM row frame00
            |-- SIM row frame01
            :
            |-- SIM row frameN-1
    ...
    """
    def __init__(self, root, num_sub, dim=(1024,1024), randomCropSize=(512, 512),\
                 transform=transforms.Compose([transforms.ToTensor()]), train=True):
        """
        Parameters
        ----------
        root : TYPE
            DESCRIPTION.
        num_sub : TYPE
            DESCRIPTION.
        dim : TYPE, optional
            DESCRIPTION. The default is (1024,1024).
        ramdomCropSize : TYPE, optional
            DESCRIPTION. The default is (512, 512).
        transform : TYPE
            DESCRIPTION.
        train : TYPE, optional
            DESCRIPTION. The default is True.
        Returns
        -------
        None.
        """
        
        # Populate the list with image paths for all the frame in `root`.
        framesPath = _make_dataset(root)
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root +"\n"))
            
        self.randomCropSize = randomCropSize
        self.cropX0         = dim[0] - randomCropSize[0]
        self.cropY0         = dim[1] - randomCropSize[1]
        self.root           = root
        self.transform      = transform
        self.train          = train
        self.num_sub        = num_sub
        self.framesPath     = framesPath
        
    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : TYPE
            DESCRIPTION.
        Returns
        -------
        None.
        """
        
        if (self.train):
            ### Data Augmentation ###
            # Apply random crop on the num_sub input frames
            cropX = random.randint(0, self.cropX0)
            cropY = random.randint(0, self.cropY0)
            cropArea = (cropX, cropY, cropX + self.randomCropSize[0], cropY + self.randomCropSize[1])
            # Random flip frame
            randomFrameFlip = random.randint(-1, 1)
        else:
            # Fixed settings to return same samples every epoch.
            # For validation/test sets.
            cropArea = (0, 0, self.randomCropSize[0], self.randomCropSize[1])
            randomFrameFlip = None
            
        real = torch.Tensor([])
        #real = _opencv_loader(self.framesPath[index][0], cropArea=cropArea, flipCode=randomFrameFlip)
        real=cv.imread(self.framesPath[index][0],cv.IMREAD_GRAYSCALE)
        
        #
        #
        #real = self.transform(real)
        img_np_float = real.astype(np.float32) / 65535.0
        img_real = torch.from_numpy(img_np_float).unsqueeze(0)
        #print(np.shape(real))
        #
        rowimage = torch.Tensor([])
        #rowimage = _opencv_loader(self.framesPath[index][1], cropArea=cropArea, flipCode=randomFrameFlip)
        rowimage=cv.imread(self.framesPath[index][1],cv.IMREAD_GRAYSCALE)
        #
        #rowimage = self.transform(rowimage)
        wf_np_float = rowimage.astype(np.float32) / 65535.0
        img_wf = torch.from_numpy(wf_np_float).unsqueeze(0)
        #
       # rowimage = torch.Tensor([])
        # Loop over for all frames corresponding to the `index`.
       # for rawIndex in range(self.num_sub):
       #     frameIndex = rawIndex+1
       #     # Open image using pil and augment the image.
       #     image = _opencv_loader(self.framesPath[index][frameIndex], cropArea=cropArea, flipCode=randomFrameFlip)
       #     # Apply transformation if specified.
       #     if self.transform is not None:
        #        image = self.transform(image)
       #     rowimage = torch.cat((rowimage, image), dim=0)
            
        return img_real, img_wf
    
    def __len__(self):
        """   
        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        return len(self.framesPath)
    
    def __repr__(self):
        """
        Returns
        -------
        None.
        """
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        
        return fmt_str
    
    
class DLSIM_v1(data.Dataset):
    """
    A dataloader for loading N samples arranged in this way:
        |-- clip0
            |-- source image
            -----------------
            |-- SIM row frame00
            |-- SIM row frame01
            :
            |-- SIM row frameN-1
            -----------------
            |-- SIM row range frame00
            |-- SIM row range frame01
            :
            |-- SIM row range frameN-1
        |-- clip1
            |-- source image
            -----------------
            |-- SIM row frame00
            |-- SIM row frame01
            :
            |-- SIM row frameN-1
            -----------------
            |-- SIM row range frame00
            |-- SIM row range frame01
            :
            |-- SIM row range frameN-1
        :
        :
        :
        |-- clipN
            |-- source image
            -----------------
            |-- SIM row frame00
            |-- SIM row frame01
            :
            |-- SIM row frameN-1
            -----------------
            |-- SIM row range frame00
            |-- SIM row range frame01
            :
            |-- SIM row range frameN-1
    ...
    """
    def __init__(self, root, num_sub, dim=(1024,1024), randomCropSize=(512, 512),\
                 transform=transforms.Compose([transforms.ToTensor()]), train=True):
        """
        Parameters
        ----------
        root : TYPE
            DESCRIPTION.
        num_sub : TYPE
            DESCRIPTION.
        dim : TYPE, optional
            DESCRIPTION. The default is (1024,1024).
        ramdomCropSize : TYPE, optional
            DESCRIPTION. The default is (512, 512).
        transform : TYPE
            DESCRIPTION.
        train : TYPE, optional
            DESCRIPTION. The default is True.
        Returns
        -------
        None.
        """
        
        # Populate the list with image paths for all the frame in `root`.
        framesPath = _make_dataset(root)
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root +"\n"))
            
        self.randomCropSize = randomCropSize
        self.cropX0         = dim[0] - randomCropSize[0]
        self.cropY0         = dim[1] - randomCropSize[1]
        self.root           = root
        self.transform      = transform
        self.train          = train
        self.num_sub        = num_sub
        self.framesPath     = framesPath
        
    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : TYPE
            DESCRIPTION.
        Returns
        -------
        None.
        """
        
        if (self.train):
            ### Data Augmentation ###
            # Apply random crop on the num_sub input frames
            cropX = random.randint(0, self.cropX0)
            cropY = random.randint(0, self.cropY0)
            cropArea = (cropX, cropY, cropX + self.randomCropSize[0], cropY + self.randomCropSize[1])
            # Random flip frame
            randomFrameFlip = random.randint(-1, 1)
        else:
            # Fixed settings to return same samples every epoch.
            # For validation/test sets.
            cropArea = (0, 0, self.randomCropSize[0], self.randomCropSize[1])
            randomFrameFlip = None
            
        real = torch.Tensor([])
        real = _opencv_loader(self.framesPath[index][0], cropArea=cropArea, flipCode=randomFrameFlip)
        real = self.transform(real)
        
        rowImage = torch.Tensor([])
        # Loop over for all frames corresponding to the `index`.
        for rawIndex in range(self.num_sub):
            frameIndex = rawIndex+1
            # Open image using pil and augment the image.
            image = _opencv_loader(self.framesPath[index][frameIndex], cropArea=cropArea, flipCode=randomFrameFlip)
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            rowImage = torch.cat((rowImage, image), dim=0)
            
        rowRange = torch.Tensor([])
        # Loop over for all frames corresponding to the `index`.
        for rawIndex in range(self.num_sub):
            frameIndex = rawIndex+1+self.num_sub
            # Open image using pil and augment the image.
            image = _opencv_loader(self.framesPath[index][frameIndex], cropArea=cropArea, flipCode=randomFrameFlip)
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            rowRange = torch.cat((rowRange, image), dim=0)
        for rawIndex in range(self.num_sub):
            frameIndex=2*self.num_sub+1
            WF_image = _opencv_loader(self.framesPath[index][frameIndex], cropArea=cropArea, flipCode=randomFrameFlip)
            if self.transform is not None:
                WF_image = self.transform(WF_image)
        return real, rowImage, rowRange,WF_image
    
    def __len__(self):
        """   
        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        return len(self.framesPath)
    
    def __repr__(self):
        """
        Returns
        -------
        None.
        """
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        
        return fmt_str