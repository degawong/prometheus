'''
Author: your name
Date: 2021-09-08 17:12:26
LastEditTime: 2021-09-09 17:22:29
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \prometheus\networks\pytorch\vgg.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import models
from collections import namedtuple

class Vgg16():
    def __init__(self) -> None:
        self.__network = models.vgg16()

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(vgg19, self).__init__()
        vgg_pretrained_features = models.vgg.vgg19(pretrained=True).features
        self.__slice1 = nn.Sequential()
        self.__slice2 = nn.Sequential()
        self.__slice3 = nn.Sequential()
        self.__slice4 = nn.Sequential()
        self.__slice5 = nn.Sequential()
        for _ in range(4):
            self.__slice1.add_module(str(_), vgg_pretrained_features[_])
        for _ in range(4, 9):
            self.__slice2.add_module(str(_), vgg_pretrained_features[_])
        for _ in range(9, 18):
            self.__slice3.add_module(str(_), vgg_pretrained_features[_])
        for _ in range(18, 27):
            self.__slice4.add_module(str(_), vgg_pretrained_features[_])
        for _ in range(27, 36):
            self.__slice5.add_module(str(_), vgg_pretrained_features[_])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        h = self.__slice1(input)
        h_relu1_2 = h
        h = self.__slice2(h)
        h_relu2_2 = h
        h = self.__slice3(h)
        h_relu3_4 = h
        h = self.__slice4(h)
        h_relu4_4 = h
        h = self.__slice5(h)
        h_relu5_4 = h
        vgg_outputs = namedtuple(
            "VggOutputs",
            [
                'relu1_2',
                'relu2_2',
                'relu3_4',
                'relu4_4',
                'relu5_4'
            ]
        )
        output = vgg_outputs(
            h_relu1_2,
            h_relu2_2,
            h_relu3_4,
            h_relu4_4,
            h_relu5_4
        )
        return output

if __name__ == '__main__':
    pass