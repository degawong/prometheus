'''
Author: your name
Date: 2021-09-09 17:23:02
LastEditTime: 2021-09-10 10:00:01
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \prometheus\networks\pytorch\residualnet.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import sys
sys.path.append('attention')

print(sys.path)
print(os.getcwd())

import pytorch.cbam as cbam
import pytorch.senet as senet

class Conv2dLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None, act=None, bias=False):
        super(Conv2dLayer, self).__init__()
        # use default padding value or (kernel size // 2) * dilation value
        if padding is not None:
            padding = padding
        else:
            padding = dilation * (kernel_size - 1) // 2
        if norm is not None:
            self.add_module('norm', norm(out_channels))
        if act is not None:
            self.add_module('act', act)
        self.add_module('conv2d', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias))

class ResidualNet(nn.Module):
    def __init__(self, channel, norm=nn.BatchNorm2d, dilation=1, bias=False, which_attention=None, reduction=None, scale=1, act=nn.ReLU(True)):
        super(ResidualNet, self).__init__()
        assert which_attention in [None, 'cbam', 'senet'], "attention should be cmam or senet"
        self.__attention_package = {
            'cbam' : cbam.CBAM,
            'senet' : senet.SeNet
        }
        self.__conv_1 = Conv2dLayer(channel, channel, kernel_size = 3, stride = 1, dilation = dilation, norm = norm, act = act, bias = bias)
        self.__conv_2 = Conv2dLayer(channel, channel, kernel_size = 3, stride = 1, dilation = dilation, norm = norm, act = None, bias = None)
        self.__scale = scale
        self.__attention_layer = None
        if which_attention is not None:
            self.__attention_layer = self.__attention_package[which_attention](channel, reduction)

    def forward(self, x):
        res = x
        x = self.__conv_1(x)
        x = self.__conv_2(x)
        if self.__attention_layer:
            x = self.__attention_layer(x)
        x = x * self.__scale
        out = x + res
        return out

if __name__ == '__main__':
    print('test in test_network')