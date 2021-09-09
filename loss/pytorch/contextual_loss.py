'''
Author: your name
Date: 2021-09-09 08:45:53
LastEditTime: 2021-09-09 10:22:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \prometheus\loss\pytorch\contextual_loss.py
'''
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append("networks")

import pytorch.vgg as vgg

'''
reference https://github.com/HaolyShiit/contextual_loss_pytorch
'''

def relative_distance(dist_raw):
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-5)
    return dist_tilde

def cosine_distance(x, y):
    # mean shifting by channel-wise mean of `y`.
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mu
    y_centered = y - y_mu
    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)
    # channel-wise vectorization
    n, c, *_ = x.size()
    x_normalized = x_normalized.reshape(n, c, -1)  # (n, c, h * w)
    y_normalized = y_normalized.reshape(n, c, -1)  # (n, c, h * w)
    # consine similarity
    cosine_sim = torch.bmm(
        x_normalized.transpose(1, 2),
        y_normalized
    )  # (n, h * w, h * w)
    # convert to distance
    dist = 1 - cosine_sim
    return dist

def l1_distance(x: torch.Tensor, y: torch.Tensor):
    (n, c, h, w) = x.size()
    x_vec = x.view(n, c, -1)
    y_vec = y.view(n, c, -1)

    dist = x_vec.unsqueeze(2) - y_vec.unsqueeze(3)
    dist = dist.sum(dim=1).abs()
    dist = dist.transpose(1, 2).reshape(n, h * w, h * w)
    dist = dist.clamp(min=0.)

    return dist

def l2_distance(x, y):
    (n, c, h, w) = x.size()
    x_vec = x.view(n, c, -1)
    y_vec = y.view(n, c, -1)
    x_s = torch.sum(x_vec ** 2, dim=1)
    y_s = torch.sum(y_vec ** 2, dim=1)

    A = y_vec.transpose(1, 2) @ x_vec
    dist = y_s - 2 * A + x_s.transpose(0, 1)
    dist = dist.transpose(1, 2).reshape(n, h * w, h * w)
    dist = dist.clamp(min=0.)

    return dist

def compute_meshgrid(shape):
    (n, c, h, w) = shape
    rows = torch.arange(0, h, dtype = torch.float32) / (h + 1)
    cols = torch.arange(0, w, dtype = torch.float32) / (w + 1)
    feature_grid = torch.meshgrid(rows, cols)
    feature_grid = torch.stack(feature_grid).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(n)], dim=0)
    return feature_grid

def compute_cx(dist_tilde, band_width):
    w = torch.exp((1 - dist_tilde) / band_width)      # Eq(3)
    cx = w / torch.sum(w, dim = 2, keepdim = True)    # Eq(4)
    return cx

def contextual_loss(
        x: torch.Tensor,
        y: torch.Tensor,
        band_width: float = 0.5,
        loss_type: str = 'cosine'
    ):
    """
    parameters
    ---
    x : torch.Tensor features of shape (N, C, H, W).
    y : torch.Tensor features of shape (N, C, H, W).
    band_width :
        float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type :
        string, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.
    returns
    ---
    contextual_loss : torch.Tensor contextual loss between x and y
    """
    assert x.size() == y.size(), 'input tensor must have the same size.'
    assert loss_type in ['l1', 'l2', 'cosine'], "select a loss type from 'l1', 'l2', 'cosine'"
    (n, c, h, w) = x.size()
    if loss_type == 'l1': raw_distance = l1_distance(x, y)
    if loss_type == 'l2': raw_distance = l2_distance(x, y)
    if loss_type == 'cosine': raw_distance = cosine_distance(x, y)
    tilde_distance = relative_distance(raw_distance)
    cx = compute_cx(tilde_distance, band_width)
    cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)
    cx_loss = torch.mean(-torch.log(cx + 1e-5))
    return cx_loss

def contextual_bilateral_loss(
        x: torch.Tensor,
        y: torch.Tensor,
        weight_sp: float = 0.1,
        band_width: float = 1.,
        loss_type: str = 'cosine'
    ):
    """
    parameters
    ---
    x : torch.Tensor features of shape (N, C, H, W).
    y : torch.Tensor features of shape (N, C, H, W).
    band_width :
        float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type :
        string, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.
    returns
    ---
    contextual_loss : torch.Tensor contextual loss between x and y
    k_arg_max_NC : torch.Tensor indices to maximize similarity over channels.
    """
    # spatial loss
    grid = compute_meshgrid(x.shape).to(x.device)
    dist_raw = l2_distance(grid, grid)
    dist_tilde = relative_distance(dist_raw)
    cx_sp = compute_cx(dist_tilde, band_width)
    # feature loss
    if loss_type == 'l1': raw_distance = l1_distance(x, y)
    if loss_type == 'l2': raw_distance = l2_distance(x, y)
    if loss_type == 'cosine': raw_distance = cosine_distance(x, y)
    tilde_distance = relative_distance(raw_distance)
    cx_feat = compute_cx(tilde_distance, band_width)
    # combined loss
    cx_combine = (1. - weight_sp) * cx_feat + weight_sp * cx_sp
    k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)
    cx = k_max_NC.mean(dim=1)
    cb_loss = torch.mean(-torch.log(cx + 1e-5))
    return cb_loss

class contextual(nn.Module):
    def __init__(self):
        super().__init__()

class contextual_bilateral(nn.Module):
    def __init__(self):
        super().__init__()

class ContextualLoss(nn.Module):
    """
    Creates a criterion that measures the contextual loss.

    Parameters
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self,
                 band_width: float = 0.5,
                 loss_type: str = 'cosine',
                 use_vgg: bool = False,
                 vgg_layer: str = 'relu3_4'):

        super(ContextualLoss, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        self.__loss_type = loss_type
        self.__band_width = band_width

        if use_vgg:
            self.vgg_model = vgg.vgg19()
            self.vgg_layer = vgg_layer
            self.register_buffer(
                name='vgg_mean',
                tensor=torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
            )
            self.register_buffer(
                name='vgg_std',
                tensor=torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
            )

    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3, 'VGG model takes 3 chennel images.'
            # normalization
            x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            # picking up vgg feature maps
            x = getattr(self.vgg_model(x), self.vgg_layer)
            y = getattr(self.vgg_model(y), self.vgg_layer)
        return contextual_loss(x, y, self.__loss_type, self.__band_width)

class ContextualBilateralLoss(nn.Module):
    """
    Creates a criterion that measures the contextual bilateral loss.

    Parameters
    ---
    weight_sp : float, optional
        a balancing weight between spatial and feature loss.
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(
            self,
            weight_sp: float = 0.1,
            band_width: float = 0.5,
            loss_type: str = 'cosine',
            use_vgg: bool = False,
            vgg_layer: str = 'relu3_4'
        ):
        super(ContextualBilateralLoss, self).__init__()
        assert band_width > 0, 'band_width parameter must be positive.'
        self.__band_width = band_width
        if use_vgg:
            self.vgg_model = vgg.vgg19()
            self.vgg_layer = vgg_layer
            self.register_buffer(
                name='vgg_mean',
                tensor=torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
            )
            self.register_buffer(
                name='vgg_std',
                tensor=torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
            )

    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3, 'VGG model takes 3 chennel images.'
            # normalization
            x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            # picking up vgg feature maps
            x = getattr(self.vgg_model(x), self.vgg_layer)
            y = getattr(self.vgg_model(y), self.vgg_layer)
        return contextual_bilateral_loss(x, y, self.__band_width)

if __name__ == '__main__':
    pass