'''
Author: your name
Date: 2021-09-08 14:43:39
LastEditTime: 2021-09-08 16:58:08
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \prometheus\attention\pytorch\se_net.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class se_net(nn.Module):
    """
    reference https://github.com/moskomule/senet.pytorch
    """
    def __init__(self, channels, reduction=16):
        super(se_net, self).__init__()
        assert (channels // reduction) > 0, "channels  // reduction should bigger than zero"
        self.__avg_pool = nn.AdaptiveAvgPool2d(1)
        self.__fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        n, c, _, _ = input.size()
        t = self.__avg_pool(input).view(n, c)
        t = self.__fc(t).view(n, c, 1, 1)
        return input * t.expand_as(input)

if __name__ == '__main__':
    torch.manual_seed(seed = 2021)
    data_in = torch.randn(1, 3, 256, 256)
    se = se_net(3, 1)
    data_out = se(data_in)
    print(data_in)
    print(data_out)
    
    
    
    
    
    
    
