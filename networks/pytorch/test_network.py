'''
Author: your name
Date: 2021-09-10 09:21:31
LastEditTime: 2021-09-10 09:28:53
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \prometheus\networks\pytorch\test_network.py
'''

import sys
sys.path.append('network/pytorch')

import pytest

import residualnet as resnet

class TestNetwork():
    def test_residualnet(self):
        resnet.ResidualNet(2, reduction = 1, which_attention = 'cbam')
        resnet.ResidualNet(2, reduction = 1, which_attention = 'senet')