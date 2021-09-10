'''
Author: your name
Date: 2021-09-09 17:11:28
LastEditTime: 2021-09-10 10:14:09
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \prometheus\algorithm\pytorch\image_inpainting\algorithm.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytest
import numpy as np

class Laplacian(nn.Module):
    def __init__(self, dim=3):
        super(Laplacian, self).__init__()
        # 2D laplacian kernel (2D LOG operator.).
        self.__channel_dim = dim
        laplacian_kernel = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])
        laplacian_kernel = np.repeat(laplacian_kernel[None, None, :, :], dim, 0)
        # learnable kernel.
        self.__kernel = torch.nn.Parameter(torch.FloatTensor(laplacian_kernel))

    def forward(self, x):
        # pyramid module in 4 scales.
        lap = F.conv2d(x, self.__kernel, groups = self.__channel_dim, padding = 1, stride = 1, dilation = 1)
        return lap

class LaplacianPyramid(nn.Module):
    # filter laplacian LOG kernel, kernel size: 3.
    # The laplacian Pyramid is used to generate high frequency images.
    def __init__(self, dim=3):
        super(LaplacianPyramid, self).__init__()
        # 2D laplacian kernel (2D LOG operator).
        self.__channel_dim = dim
        laplacian_kernel = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])
        laplacian_kernel = np.repeat(laplacian_kernel[None, None, :, :], dim, 0)
        # learnable laplacian kernel
        self.__kernel = torch.nn.Parameter(torch.FloatTensor(laplacian_kernel))

    def forward(self, x):
        # pyramid module for 4 scales.
        x0 = F.interpolate(x, scale_factor=0.125, mode='bilinear')
        x1 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        x2 = F.interpolate(x, scale_factor=0.50, mode='bilinear')
        lap_0 = F.conv2d(x0, self.__kernel, groups=self.__channel_dim, padding=1, stride=1, dilation=1)
        lap_1 = F.conv2d(x1, self.__kernel, groups=self.__channel_dim, padding=1, stride=1, dilation=1)
        lap_2 = F.conv2d(x2, self.__kernel, groups=self.__channel_dim, padding=1, stride=1, dilation=1)
        lap_3 = F.conv2d(x, self.__kernel, groups=self.__channel_dim, padding=1, stride=1, dilation=1)
        lap_0 = F.interpolate(lap_0, scale_factor=8, mode='bilinear')
        lap_1 = F.interpolate(lap_1, scale_factor=4, mode='bilinear')
        lap_2 = F.interpolate(lap_2, scale_factor=2, mode='bilinear')
        return torch.cat([lap_0, lap_1, lap_2, lap_3], 1)
