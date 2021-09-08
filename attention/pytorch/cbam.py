'''
Author: your name
Date: 2021-09-08 15:27:06
LastEditTime: 2021-09-08 16:58:02
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \prometheus\attention\pytorch\cbam.py
'''

import torch
from torch import nn
"""
reference https://github.com/Jongchan/attention-module
"""
class channel_attention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(channel_attention, self).__init__()
        assert (in_channels // reduction) > 0, "channels  // reduction should bigger than zero"
        self.__relu1 = nn.ReLU()
        self.__sigmoid = nn.Sigmoid()
        self.__avg_pool = nn.AdaptiveAvgPool2d(1)
        self.__max_pool = nn.AdaptiveMaxPool2d(1)
        self.__fc_1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.__fc_2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)

    def forward(self, input):
        avg_out = self.__fc_2(self.__relu1(self.__fc_1(self.__avg_pool(input))))
        max_out = self.__fc_2(self.__relu1(self.__fc_1(self.__max_pool(input))))
        return self.__sigmoid(avg_out + max_out)

class spatial_attention(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attention, self).__init__()
        padding = (kernel_size - 1) // 2
        assert kernel_size in (3, 5, 7), 'kernel size must be 3 5 7'
        self.__sigmoid = nn.Sigmoid()
        self.__conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, input):
        avg_out = torch.mean(input, dim=1, keepdim=True)
        max_out, _ = torch.max(input, dim=1, keepdim=True)
        return self.__sigmoid(self.__conv(torch.cat([avg_out, max_out], dim=1)))

class cbam(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(cbam, self).__init__()
        self.__ca = channel_attention(in_channels, reduction)
        self.__sa = spatial_attention(kernel_size)

    def forward(self, input):
        output = input * self.__ca(input)
        return output * self.__sa(output)

if __name__ == '__main__':
    torch.manual_seed(seed=20200910)
    ca = channel_attention(32)
    data_in = torch.randn(8, 32, 300, 300)
    data_out = ca(data_in)

if __name__ == '__main__':
    torch.manual_seed(seed=20200910)
    sa = spatial_attention(3)
    data_in = torch.randn(8, 32, 300, 300)
    data_out = sa(data_in)

if __name__ == '__main__':
    torch.manual_seed(seed=20200910)
    cs = cbam(32, 16, 3)
    data_in = torch.randn(8, 32, 300, 300)
    data_out = cs(data_in)
