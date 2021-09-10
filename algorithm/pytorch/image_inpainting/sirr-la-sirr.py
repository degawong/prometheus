'''
Author: your name
Date: 2021-09-10 09:36:23
LastEditTime: 2021-09-10 15:22:00
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \prometheus\algorithm\pytorch\image_inpainting\sirr-la-sirr.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys

sys.path.extend(['networks', 'attention'])

import collections

import algorithm as pack
import pytorch.cbam as cbam
import pytorch.residualnet as resnet

class LRM(nn.Module):
    def __init__(self, device):
        super(LRM, self).__init__()
        # multi-scale laplacian submodules (RDMs)
        # self.lap_single = SingleLaplacian(device, dim=6)
        self.__conv_00 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv2d', nn.Conv2d(6, 32, 3, 1, 1)),
                    ('relu', nn.ReLU()),
                ]
            )
        )
        # SE-resblocks(ReLU)
        self.__se_resblock_package_relu = nn.Sequential(
            collections.OrderedDict(
                [
                    ('layer-00', resnet.ResidualNet(32, norm=None, which_attention = 'senet', reduction = 2, scale = 0.1)),
                    ('relu-01', nn.ReLU()),
                    ('layer-02', resnet.ResidualNet(32, norm=None, which_attention = 'senet', reduction = 2, scale = 0.1)),
                    ('relu-03', nn.ReLU()),
                    ('layer-04', resnet.ResidualNet(32, norm=None, which_attention = 'senet', reduction = 2, scale = 0.1)),
                    ('relu-05', nn.ReLU()),
                    ('layer-06', resnet.ResidualNet(32, norm=None, which_attention = 'senet', reduction = 2, scale = 0.1)),
                    ('relu-07', nn.ReLU()),
                    ('layer-08', resnet.ResidualNet(32, norm=None, which_attention = 'senet', reduction = 2, scale = 0.1)),
                    ('relu-09', nn.ReLU()),
                    ('layer-10', resnet.ResidualNet(32, norm=None, which_attention = 'senet', reduction = 2, scale = 0.1)),
                    ('relu-11', nn.ReLU()),
                ]
            ) 
        )
        # Laplacian blocks 
        self.__laplace_pyramid = pack.LaplacianPyramid(dim = 6)
        # Convolutional blocks for encoding laplacian features. 
        self.__conv_01 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv2d', nn.Conv2d(24, 32, 3, 1, 1)),
                    ('relu', nn.PReLU()),
                ]
            )
        )
        # SE-resblocks(P-ReLU)
        self.__se_resblock_package_prelu_01 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('layer-00', resnet.ResidualNet(32, norm=None, which_attention = 'senet', reduction = 2, scale = 0.1, act=nn.PReLU())),
                    ('prelu-01', nn.PReLU()),
                    ('layer-02', resnet.ResidualNet(32, norm=None, which_attention = 'senet', reduction = 2, scale = 0.1, act=nn.PReLU())),
                    ('prelu-03', nn.PReLU()),
                    ('layer-04', resnet.ResidualNet(32, norm=None, which_attention = 'senet', reduction = 2, scale = 0.1, act=nn.PReLU())),
                    ('prelu-05', nn.PReLU()),
                ]
            ) 
        )
        self.__se_resblock_package_prelu_02 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('layer-00', resnet.ResidualNet(32, norm=None, which_attention = 'senet', reduction = 2, scale = 0.1, act=nn.PReLU())),
                    ('prelu-01', nn.PReLU()),
                    ('layer-02', resnet.ResidualNet(32, norm=None, which_attention = 'senet', reduction = 2, scale = 0.1, act=nn.PReLU())),
                    ('prelu-03', nn.PReLU()),
                    ('layer-04', resnet.ResidualNet(32, norm=None, which_attention = 'senet', reduction = 2, scale = 0.1, act=nn.PReLU())),
                    ('prelu-05', nn.PReLU()),
                ]
            )
        )

        # Convolutional block for RCMap_{i+1}
        self.__conv_02 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv-00', nn.Conv2d(32, 32, 3, 1, 1)),
                    ('relu-01', nn.ReLU()),
                    ('conv-02', nn.Conv2d(32, 1, 3, 1, 1)),
                    ('sigmoid-03', nn.Sigmoid()),
                ]
            )
        )

        # LSTM block.
        self.__lstm_conv_sigmoid_01 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv2d', nn.Conv2d(32 * 4, 32 * 2, 3, 1, 1)),
                    ('sigmoid', nn.Sigmoid()),
                ]
            )
        )
        self.__lstm_conv_sigmoid_02 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv2d', nn.Conv2d(32 * 4, 32 * 2, 3, 1, 1)),
                    ('sigmoid', nn.Sigmoid()),
                ]
            )
        )
        self.__lstm_conv_tanh = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv2d', nn.Conv2d(32 * 4, 32 * 2, 3, 1, 1)),
                    ('tanh', nn.Tanh()),
                ]
            )
        )
        self.__lstm_conv_sigmoid_03 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv2d', nn.Conv2d(32 * 4, 32 * 2, 3, 1, 1)),
                    ('sigmoid', nn.Sigmoid()),
                ]
            )
        )

        # Convolutional block for R_{i+1}
        self.__conv_03 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv-00', nn.Conv2d(32 * 2, 32, 3, 1, 1)),
                    ('relu-01', nn.ReLU()),
                    ('conv-02', nn.Conv2d(32, 3, 3, 1, 1)),
                    ('relu-03', nn.ReLU()),
                ]
            )
        )

        # Auto-Encoder.
        self.__autoencoder_block_01 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv-00', nn.Conv2d(10, 64, 5, 1, 2)),
                    ('relu-01', nn.ReLU()),
                    ('cbam-02', resnet.ResidualNet(64, which_attention = 'cbam', reduction = 2)),
                ]
            )
        )
        self.__autoencoder_block_02 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv-00', nn.Conv2d(64, 128, 3, 2, 1)),
                    ('relu-01', nn.ReLU()),
                    ('conv-02', nn.Conv2d(128, 128, 3, 1, 1)),
                    ('relu-03', nn.ReLU()),
                    ('cbam-04', resnet.ResidualNet(128, which_attention = 'cbam', reduction = 4)),
                    ('cbam-05', resnet.ResidualNet(128, which_attention = 'cbam', reduction = 4)),
                ]
            )
        )
        self.__autoencoder_block_03 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv-00', nn.Conv2d(128, 256, 3, 2, 1)),
                    ('relu-01', nn.ReLU()),
                    ('conv-02', nn.Conv2d(256, 256, 3, 1, 1)),
                    ('relu-03', nn.ReLU()),
                    ('conv-04', nn.Conv2d(256, 256, 3, 1, 1)),
                    ('relu-05', nn.ReLU()),
                    ('cbam-06', resnet.ResidualNet(256, which_attention = 'cbam', reduction = 8)),
                    ('cbam-07', resnet.ResidualNet(256, which_attention = 'cbam', reduction = 8)),
                    ('dilation-conv-08', nn.Conv2d(256, 256, 3, 1, 2, dilation = 2)),
                    ('relu-09', nn.ReLU()),
                    ('dilation-conv-10', nn.Conv2d(256, 256, 3, 1, 4, dilation = 4)),
                    ('relu-11', nn.ReLU()),
                    ('dilation-conv-12', nn.Conv2d(256, 256, 3, 1, 8, dilation = 8)),
                    ('relu-13', nn.ReLU()),
                    ('dilation-conv-14', nn.Conv2d(256, 256, 3, 1, 16, dilation = 16)),
                    ('relu-15', nn.ReLU()),
                    ('conv-16', nn.Conv2d(256, 256, 3, 1, 1)),
                    ('relu-17', nn.ReLU()),
                    ('conv-18', nn.Conv2d(256, 256, 3, 1, 1)),
                    ('relu-19', nn.ReLU()),
                ]
            )
        )
        self.__deconv_01 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('transpose-conv-00', nn.ConvTranspose2d(256, 128, 4, 2, 1)),
                    ('padding-01', nn.ReflectionPad2d((1, 0, 1, 0))),
                    ('average-pool-02', nn.AvgPool2d(2, stride = 1)),
                    ('relu-03', nn.ReLU()),
                ]
            )
        )
        self.__deconv_02 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('transpose-conv-00', nn.ConvTranspose2d(128, 64, 4, 2, 1)),
                    ('padding-01', nn.ReflectionPad2d((1, 0, 1, 0))),
                    ('average-pool-02', nn.AvgPool2d(2, stride = 1)),
                    ('relu-03', nn.ReLU()),
                ]
            )
        )
        self.__conv_04 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv-00', nn.Conv2d(128, 128, 3, 1, 1)),
                    ('relu-01', nn.ReLU()),
                ]
            )
        )
        self.__conv_05 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv-00', nn.Conv2d(64, 32, 3, 1, 1)),
                    ('relu-01', nn.ReLU()),
                ]
            )
        )
        self.__frame_01_04 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv-00', nn.Conv2d(256, 3, 3, 1, 1)),
                    ('relu-01', nn.ReLU()),
                ]
            )
        )
        self.__frame_01_02 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv-00', nn.Conv2d(128, 3, 3, 1, 1)),
                    ('relu-01', nn.ReLU()),
                ]
            )
        )
        self.__output = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv-00', nn.Conv2d(32, 3, 3, 1, 1)),
                    ('relu-01', nn.ReLU()),
                ]
            )
        )

    def forward(self, I, T, h, c):
        # I: original image.
        # T: transmission image.
        # h, c: hidden states for LSTM block in stage 1.

        x = torch.cat([I, T], 1)
        # get laplacian(frequency) information of [I,T].
        laplace_layer = self.__laplace_pyramid(x)

        # ----- stage 1 -----
        # encode [I, T].
        x = self.__conv_00(x)
        x = self.det_conv4_2__se_resblock_package_relu(x)

        # encode [I_lap, T_lap].
        laplace_layer = self.__conv_01(laplace_layer)
        # se-resblock layer3 for [I_lap, T_lap] features (p-relu for activation.)
        laplace_layer = self.__se_resblock_package_prelu_01(laplace_layer)
        # predict RCMap from laplacian features.
        c_map = self.__conv_02(laplace_layer)
        # se-resblock layer4 for [I_lap, T_lap] features (p-relu for activation.)
        laplace_layer = self.__se_resblock_package_prelu_02(laplace_layer)
        # suppress transmission features.
        laplace_layer = (1 - c_map) * laplace_layer

        # concat image & laplacian feature and recurrent features.
        x = torch.cat([x, laplace_layer, h], 1)

        # lstm.
        i = self.__lstm_conv_sigmoid_01(x)
        f = self.__lstm_conv_sigmoid_02(x)
        g = self.__lstm_conv_tanh(x)
        o = self.__lstm_conv_sigmoid_03(x)
        c = f * c + i * g
        h = o * torch.tanh(c)
        reflect = self.__conv_03(h)

        # ------ stage2 ------ 
        # predict T_{i+1} with input: R_{i+1}, T_i, C_{i+1}.
        x = torch.cat([I, T, reflect, c_map], 1)
        x = self.__autoencoder_block_01(x)
        res_1 = x
        x = self.__autoencoder_block_02(x)
        res_2 = x
        x = self.__autoencoder_block_03(x)

        frame_1 = self.__frame_01_04(x)
        x = self.__deconv_01(x)
        x = x + res_2
        x = self.__conv_04(x)
        frame_2 = self.__frame_01_02(x)
        x = self.__deconv_02(x)
        x = x + res_1
        x = self.__conv_05(x)
        x = self.__output(x)

        return h, c, c_map, reflect, frame_1, frame_2, x

if __name__ == '__main__':
    print('test in test algorithm')