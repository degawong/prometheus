'''
Author: your name
Date: 2021-09-08 08:58:50
LastEditTime: 2021-09-08 16:57:37
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \prometheus\networks\pytorch\unet.py
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class u_net(nn.Module):
    """
    this class mainly for personal usage
    """
    def __init__(self, in_channels, out_channels, n_feats, norm=nn.BatchNorm2d, se_reduction=None, res_scale=1, transposedConv=False, res_block=True) -> None:
        super().__init__()
    class __conv_block(nn.Sequential):
        def __init__(self, operation, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, norm=None, act=None) -> None:
            super().__init__()
            padding = padding or dilation * (kernel_size - 1) // 2
            self.add_module('padding', nn.ReflectionPad2d(padding)) if padding > 1 else None
            self.add_module('conv2d', operation(in_channels, out_channels, kernel_size, stride, padding=1 if padding <= 1 else padding, dilation=dilation))
            self.add_module('norm', norm(out_channels)) if norm is not None else None
            self.add_module('act', act(inplace=True)) if act is not None else None

if __name__ == '__main__':
    pass