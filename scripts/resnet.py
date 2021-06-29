import os
import sys
import math
from time import time
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn.functional as functional
from inplace_abn import ABN, InPlaceABN, InPlaceABNSync

from hlaspp import ASPP
from hlconv import *
from decoder import *
from BilinearModel import *
# import models
# from models.util import try_index
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

model_urls = {
    'resnet34': 'https://cloudstor.aarnet.edu.au/plus/s/zcfoVwqiUIKtzoJ/download',
}

CORRESP_NAME = {
    "module.mod1.conv1.weight": "layer0.0.weight",
    "module.mod1.bn1.weight": "layer0.1.weight",
    "module.mod1.bn1.bias": "layer0.1.bias",
    "module.mod1.bn1.running_mean": "layer0.1.running_mean",
    "module.mod1.bn1.running_var": "layer0.1.running_var",
    "module.mod2.block1.convs.conv1.weight": "layer1.block1.convs.conv1.weight",
    "module.mod2.block1.convs.bn1.weight": "layer1.block1.convs.bn1.weight",
    "module.mod2.block1.convs.bn1.bias": "layer1.block1.convs.bn1.bias",
    "module.mod2.block1.convs.bn1.running_mean": "layer1.block1.convs.bn1.running_mean",
    "module.mod2.block1.convs.bn1.running_var": "layer1.block1.convs.bn1.running_var",
    "module.mod2.block1.convs.conv2.weight": "layer1.block1.convs.conv2.weight",
    "module.mod2.block1.convs.bn2.weight": "layer1.block1.convs.bn2.weight",
    "module.mod2.block1.convs.bn2.bias": "layer1.block1.convs.bn2.bias",
    "module.mod2.block1.convs.bn2.running_mean": "layer1.block1.convs.bn2.running_mean",
    "module.mod2.block1.convs.bn2.running_var": "layer1.block1.convs.bn2.running_var",
    "module.mod2.block2.convs.conv1.weight": "layer1.block2.convs.conv1.weight",
    "module.mod2.block2.convs.bn1.weight": "layer1.block2.convs.bn1.weight",
    "module.mod2.block2.convs.bn1.bias": "layer1.block2.convs.bn1.bias",
    "module.mod2.block2.convs.bn1.running_mean": "layer1.block2.convs.bn1.running_mean",
    "module.mod2.block2.convs.bn1.running_var": "layer1.block2.convs.bn1.running_var",
    "module.mod2.block2.convs.conv2.weight": "layer1.block2.convs.conv2.weight",
    "module.mod2.block2.convs.bn2.weight": "layer1.block2.convs.bn2.weight",
    "module.mod2.block2.convs.bn2.bias": "layer1.block2.convs.bn2.bias",
    "module.mod2.block2.convs.bn2.running_mean": "layer1.block2.convs.bn2.running_mean",
    "module.mod2.block2.convs.bn2.running_var": "layer1.block2.convs.bn2.running_var",
    "module.mod2.block3.convs.conv1.weight": "layer1.block3.convs.conv1.weight",
    "module.mod2.block3.convs.bn1.weight": "layer1.block3.convs.bn1.weight",
    "module.mod2.block3.convs.bn1.bias": "layer1.block3.convs.bn1.bias",
    "module.mod2.block3.convs.bn1.running_mean": "layer1.block3.convs.bn1.running_mean",
    "module.mod2.block3.convs.bn1.running_var": "layer1.block3.convs.bn1.running_var",
    "module.mod2.block3.convs.conv2.weight": "layer1.block3.convs.conv2.weight",
    "module.mod2.block3.convs.bn2.weight": "layer1.block3.convs.bn2.weight",
    "module.mod2.block3.convs.bn2.bias": "layer1.block3.convs.bn2.bias",
    "module.mod2.block3.convs.bn2.running_mean": "layer1.block3.convs.bn2.running_mean",
    "module.mod2.block3.convs.bn2.running_var": "layer1.block3.convs.bn2.running_var",
    "module.mod3.block1.convs.conv1.weight": "layer2_1.layer1.conv1.weight",
    "module.mod3.block1.convs.bn1.weight": "layer2_1.layer1.bn1.weight",
    "module.mod3.block1.convs.bn1.bias": "layer2_1.layer1.bn1.bias",
    "module.mod3.block1.convs.bn1.running_mean": "layer2_1.layer1.bn1.running_mean",
    "module.mod3.block1.convs.bn1.running_var": "layer2_1.layer1.bn1.running_var",
    "module.mod3.block1.convs.conv2.weight": "layer2_1.layer2.conv2.weight",
    "module.mod3.block1.convs.bn2.weight": "layer2_1.layer2.bn2.weight",
    "module.mod3.block1.convs.bn2.bias": "layer2_1.layer2.bn2.bias",
    "module.mod3.block1.convs.bn2.running_mean": "layer2_1.layer2.bn2.running_mean",
    "module.mod3.block1.convs.bn2.running_var": "layer2_1.layer2.bn2.running_var",
    "module.mod3.block1.proj_conv.weight": "layer2_1.proj_conv.weight",
    "module.mod3.block1.proj_bn.weight": "layer2_1.proj_bn.weight",
    "module.mod3.block1.proj_bn.bias": "layer2_1.proj_bn.bias",
    "module.mod3.block1.proj_bn.running_mean": "layer2_1.proj_bn.running_mean",
    "module.mod3.block1.proj_bn.running_var": "layer2_1.proj_bn.running_var",
    "module.mod3.block2.convs.conv1.weight": "layer2_2.block1.convs.conv1.weight",
    "module.mod3.block2.convs.bn1.weight": "layer2_2.block1.convs.bn1.weight",
    "module.mod3.block2.convs.bn1.bias": "layer2_2.block1.convs.bn1.bias",
    "module.mod3.block2.convs.bn1.running_mean": "layer2_2.block1.convs.bn1.running_mean",
    "module.mod3.block2.convs.bn1.running_var": "layer2_2.block1.convs.bn1.running_var",
    "module.mod3.block2.convs.conv2.weight": "layer2_2.block1.convs.conv2.weight",
    "module.mod3.block2.convs.bn2.weight": "layer2_2.block1.convs.bn2.weight",
    "module.mod3.block2.convs.bn2.bias": "layer2_2.block1.convs.bn2.bias",
    "module.mod3.block2.convs.bn2.running_mean": "layer2_2.block1.convs.bn2.running_mean",
    "module.mod3.block2.convs.bn2.running_var": "layer2_2.block1.convs.bn2.running_var",
    "module.mod3.block3.convs.conv1.weight": "layer2_2.block2.convs.conv1.weight",
    "module.mod3.block3.convs.bn1.weight": "layer2_2.block2.convs.bn1.weight",
    "module.mod3.block3.convs.bn1.bias": "layer2_2.block2.convs.bn1.bias",
    "module.mod3.block3.convs.bn1.running_mean": "layer2_2.block2.convs.bn1.running_mean",
    "module.mod3.block3.convs.bn1.running_var": "layer2_2.block2.convs.bn1.running_var",
    "module.mod3.block3.convs.conv2.weight": "layer2_2.block2.convs.conv2.weight",
    "module.mod3.block3.convs.bn2.weight": "layer2_2.block2.convs.bn2.weight",
    "module.mod3.block3.convs.bn2.bias": "layer2_2.block2.convs.bn2.bias",
    "module.mod3.block3.convs.bn2.running_mean": "layer2_2.block2.convs.bn2.running_mean",
    "module.mod3.block3.convs.bn2.running_var": "layer2_2.block2.convs.bn2.running_var",
    "module.mod3.block4.convs.conv1.weight": "layer2_2.block3.convs.conv1.weight",
    "module.mod3.block4.convs.bn1.weight": "layer2_2.block3.convs.bn1.weight",
    "module.mod3.block4.convs.bn1.bias": "layer2_2.block3.convs.bn1.bias",
    "module.mod3.block4.convs.bn1.running_mean": "layer2_2.block3.convs.bn1.running_mean",
    "module.mod3.block4.convs.bn1.running_var": "layer2_2.block3.convs.bn1.running_var",
    "module.mod3.block4.convs.conv2.weight": "layer2_2.block3.convs.conv2.weight",
    "module.mod3.block4.convs.bn2.weight": "layer2_2.block3.convs.bn2.weight",
    "module.mod3.block4.convs.bn2.bias": "layer2_2.block3.convs.bn2.bias",
    "module.mod3.block4.convs.bn2.running_mean": "layer2_2.block3.convs.bn2.running_mean",
    "module.mod3.block4.convs.bn2.running_var": "layer2_2.block3.convs.bn2.running_var",
    "module.mod4.block1.convs.conv1.weight": "layer3_1.layer1.conv1.weight",
    "module.mod4.block1.convs.bn1.weight": "layer3_1.layer1.bn1.weight",
    "module.mod4.block1.convs.bn1.bias": "layer3_1.layer1.bn1.bias",
    "module.mod4.block1.convs.bn1.running_mean": "layer3_1.layer1.bn1.running_mean",
    "module.mod4.block1.convs.bn1.running_var": "layer3_1.layer1.bn1.running_var",
    "module.mod4.block1.convs.conv2.weight": "layer3_1.layer2.conv2.weight",
    "module.mod4.block1.convs.bn2.weight": "layer3_1.layer2.bn2.weight",
    "module.mod4.block1.convs.bn2.bias": "layer3_1.layer2.bn2.bias",
    "module.mod4.block1.convs.bn2.running_mean": "layer3_1.layer2.bn2.running_mean",
    "module.mod4.block1.convs.bn2.running_var": "layer3_1.layer2.bn2.running_var",
    "module.mod4.block1.proj_conv.weight": "layer3_1.proj_conv.weight",
    "module.mod4.block1.proj_bn.weight": "layer3_1.proj_bn.weight",
    "module.mod4.block1.proj_bn.bias": "layer3_1.proj_bn.bias",
    "module.mod4.block1.proj_bn.running_mean": "layer3_1.proj_bn.running_mean",
    "module.mod4.block1.proj_bn.running_var": "layer3_1.proj_bn.running_var",
    "module.mod4.block2.convs.conv1.weight": "layer3_2.block1.convs.conv1.weight",
    "module.mod4.block2.convs.bn1.weight": "layer3_2.block1.convs.bn1.weight",
    "module.mod4.block2.convs.bn1.bias": "layer3_2.block1.convs.bn1.bias",
    "module.mod4.block2.convs.bn1.running_mean": "layer3_2.block1.convs.bn1.running_mean",
    "module.mod4.block2.convs.bn1.running_var": "layer3_2.block1.convs.bn1.running_var",
    "module.mod4.block2.convs.conv2.weight": "layer3_2.block1.convs.conv2.weight",
    "module.mod4.block2.convs.bn2.weight": "layer3_2.block1.convs.bn2.weight",
    "module.mod4.block2.convs.bn2.bias": "layer3_2.block1.convs.bn2.bias",
    "module.mod4.block2.convs.bn2.running_mean": "layer3_2.block1.convs.bn2.running_mean",
    "module.mod4.block2.convs.bn2.running_var": "layer3_2.block1.convs.bn2.running_var",
    "module.mod4.block3.convs.conv1.weight": "layer3_2.block2.convs.conv1.weight",
    "module.mod4.block3.convs.bn1.weight": "layer3_2.block2.convs.bn1.weight",
    "module.mod4.block3.convs.bn1.bias": "layer3_2.block2.convs.bn1.bias",
    "module.mod4.block3.convs.bn1.running_mean": "layer3_2.block2.convs.bn1.running_mean",
    "module.mod4.block3.convs.bn1.running_var": "layer3_2.block2.convs.bn1.running_var",
    "module.mod4.block3.convs.conv2.weight": "layer3_2.block2.convs.conv2.weight",
    "module.mod4.block3.convs.bn2.weight": "layer3_2.block2.convs.bn2.weight",
    "module.mod4.block3.convs.bn2.bias": "layer3_2.block2.convs.bn2.bias",
    "module.mod4.block3.convs.bn2.running_mean": "layer3_2.block2.convs.bn2.running_mean",
    "module.mod4.block3.convs.bn2.running_var": "layer3_2.block2.convs.bn2.running_var",
    "module.mod4.block4.convs.conv1.weight": "layer3_2.block3.convs.conv1.weight",
    "module.mod4.block4.convs.bn1.weight": "layer3_2.block3.convs.bn1.weight",
    "module.mod4.block4.convs.bn1.bias": "layer3_2.block3.convs.bn1.bias",
    "module.mod4.block4.convs.bn1.running_mean": "layer3_2.block3.convs.bn1.running_mean",
    "module.mod4.block4.convs.bn1.running_var": "layer3_2.block3.convs.bn1.running_var",
    "module.mod4.block4.convs.conv2.weight": "layer3_2.block3.convs.conv2.weight",
    "module.mod4.block4.convs.bn2.weight": "layer3_2.block3.convs.bn2.weight",
    "module.mod4.block4.convs.bn2.bias": "layer3_2.block3.convs.bn2.bias",
    "module.mod4.block4.convs.bn2.running_mean": "layer3_2.block3.convs.bn2.running_mean",
    "module.mod4.block4.convs.bn2.running_var": "layer3_2.block3.convs.bn2.running_var",
    "module.mod4.block5.convs.conv1.weight": "layer3_2.block4.convs.conv1.weight",
    "module.mod4.block5.convs.bn1.weight": "layer3_2.block4.convs.bn1.weight",
    "module.mod4.block5.convs.bn1.bias": "layer3_2.block4.convs.bn1.bias",
    "module.mod4.block5.convs.bn1.running_mean": "layer3_2.block4.convs.bn1.running_mean",
    "module.mod4.block5.convs.bn1.running_var": "layer3_2.block4.convs.bn1.running_var",
    "module.mod4.block5.convs.conv2.weight": "layer3_2.block4.convs.conv2.weight",
    "module.mod4.block5.convs.bn2.weight": "layer3_2.block4.convs.bn2.weight",
    "module.mod4.block5.convs.bn2.bias": "layer3_2.block4.convs.bn2.bias",
    "module.mod4.block5.convs.bn2.running_mean": "layer3_2.block4.convs.bn2.running_mean",
    "module.mod4.block5.convs.bn2.running_var": "layer3_2.block4.convs.bn2.running_var",
    "module.mod4.block6.convs.conv1.weight": "layer3_2.block5.convs.conv1.weight",
    "module.mod4.block6.convs.bn1.weight": "layer3_2.block5.convs.bn1.weight",
    "module.mod4.block6.convs.bn1.bias": "layer3_2.block5.convs.bn1.bias",
    "module.mod4.block6.convs.bn1.running_mean": "layer3_2.block5.convs.bn1.running_mean",
    "module.mod4.block6.convs.bn1.running_var": "layer3_2.block5.convs.bn1.running_var",
    "module.mod4.block6.convs.conv2.weight": "layer3_2.block5.convs.conv2.weight",
    "module.mod4.block6.convs.bn2.weight": "layer3_2.block5.convs.bn2.weight",
    "module.mod4.block6.convs.bn2.bias": "layer3_2.block5.convs.bn2.bias",
    "module.mod4.block6.convs.bn2.running_mean": "layer3_2.block5.convs.bn2.running_mean",
    "module.mod4.block6.convs.bn2.running_var": "layer3_2.block5.convs.bn2.running_var",
    "module.mod5.block1.convs.conv1.weight": "layer4_1.layer1.conv1.weight",
    "module.mod5.block1.convs.bn1.weight": "layer4_1.layer1.bn1.weight",
    "module.mod5.block1.convs.bn1.bias": "layer4_1.layer1.bn1.bias",
    "module.mod5.block1.convs.bn1.running_mean": "layer4_1.layer1.bn1.running_mean",
    "module.mod5.block1.convs.bn1.running_var": "layer4_1.layer1.bn1.running_var",
    "module.mod5.block1.convs.conv2.weight": "layer4_1.layer2.conv2.weight",
    "module.mod5.block1.convs.bn2.weight": "layer4_1.layer2.bn2.weight",
    "module.mod5.block1.convs.bn2.bias": "layer4_1.layer2.bn2.bias",
    "module.mod5.block1.convs.bn2.running_mean": "layer4_1.layer2.bn2.running_mean",
    "module.mod5.block1.convs.bn2.running_var": "layer4_1.layer2.bn2.running_var",
    "module.mod5.block1.proj_conv.weight": "layer4_1.proj_conv.weight",
    "module.mod5.block1.proj_bn.weight": "layer4_1.proj_bn.weight",
    "module.mod5.block1.proj_bn.bias": "layer4_1.proj_bn.bias",
    "module.mod5.block1.proj_bn.running_mean": "layer4_1.proj_bn.running_mean",
    "module.mod5.block1.proj_bn.running_var": "layer4_1.proj_bn.running_var",
    "module.mod5.block2.convs.conv1.weight": "layer4_2.block1.convs.conv1.weight",
    "module.mod5.block2.convs.bn1.weight": "layer4_2.block1.convs.bn1.weight",
    "module.mod5.block2.convs.bn1.bias": "layer4_2.block1.convs.bn1.bias",
    "module.mod5.block2.convs.bn1.running_mean": "layer4_2.block1.convs.bn1.running_mean",
    "module.mod5.block2.convs.bn1.running_var": "layer4_2.block1.convs.bn1.running_var",
    "module.mod5.block2.convs.conv2.weight": "layer4_2.block1.convs.conv2.weight",
    "module.mod5.block2.convs.bn2.weight": "layer4_2.block1.convs.bn2.weight",
    "module.mod5.block2.convs.bn2.bias": "layer4_2.block1.convs.bn2.bias",
    "module.mod5.block2.convs.bn2.running_mean": "layer4_2.block1.convs.bn2.running_mean",
    "module.mod5.block2.convs.bn2.running_var": "layer4_2.block1.convs.bn2.running_var",
    "module.mod5.block3.convs.conv1.weight": "layer4_2.block2.convs.conv1.weight",
    "module.mod5.block3.convs.bn1.weight": "layer4_2.block2.convs.bn1.weight",
    "module.mod5.block3.convs.bn1.bias": "layer4_2.block2.convs.bn1.bias",
    "module.mod5.block3.convs.bn1.running_mean": "layer4_2.block2.convs.bn1.running_mean",
    "module.mod5.block3.convs.bn1.running_var": "layer4_2.block2.convs.bn1.running_var",
    "module.mod5.block3.convs.conv2.weight": "layer4_2.block2.convs.conv2.weight",
    "module.mod5.block3.convs.bn2.weight": "layer4_2.block2.convs.bn2.weight",
    "module.mod5.block3.convs.bn2.bias": "layer4_2.block2.convs.bn2.bias",
    "module.mod5.block3.convs.bn2.running_mean": "layer4_2.block2.convs.bn2.running_mean",
    "module.mod5.block3.convs.bn2.running_var": "layer4_2.block2.convs.bn2.running_var",
}

def try_index(scalar_or_list, i):
    try:
        return scalar_or_list[i]
    except TypeError:
        return scalar_or_list

class ResidualBlock(nn.Module):
    """Configurable residual block

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : list of int
        Number of channels in the internal feature maps. Can either have two or three elements: if three construct
        a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
        `3 x 3` then `1 x 1` convolutions.
    stride : int
        Stride of the first `3 x 3` convolution
    dilation : int
        Dilation to apply to the `3 x 3` convolutions.
    groups : int
        Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
        bottleneck blocks.
    norm_act : callable
        Function to create normalization / activation Module.
    dropout: callable
        Function to create Dropout Module.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_act=ABN,
                 dropout=None):
        super(ResidualBlock, self).__init__()

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        if not is_bottleneck:
            bn2 = norm_act(channels[1])
            bn2.activation = "identity"
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False,
                                    dilation=dilation)),
                ("bn1", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,
                                    dilation=dilation)),
                ("bn2", bn2)
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            bn3 = norm_act(channels[2])
            bn3.activation = "identity"
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 1, stride=1, padding=0, bias=False)),
                ("bn1", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=stride, padding=dilation, bias=False,
                                    groups=groups, dilation=dilation)),
                ("bn2", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False)),
                ("bn3", bn3)
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)
            self.proj_bn = norm_act(channels[-1])
            self.proj_bn.activation = "identity"

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            residual = self.proj_conv(x)
            residual = self.proj_bn(residual)
        else:
            residual = x
        x = self.convs(x) + residual

        if self.convs.bn1.activation == "leaky_relu":
            return functional.leaky_relu(x, negative_slope=self.convs.bn1.activation_param, inplace=True)
        elif self.convs.bn1.activation == "elu":
            return functional.elu(x, alpha=self.convs.bn1.activation_param, inplace=True)
        elif self.convs.bn1.activation == "identity":
            return x



class ResidualBlock_bilinear(nn.Module):
    """Configurable residual block

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : list of int
        Number of channels in the internal feature maps. Can either have two or three elements: if three construct
        a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
        `3 x 3` then `1 x 1` convolutions.
    stride : int
        Stride of the first `3 x 3` convolution
    dilation : int
        Dilation to apply to the `3 x 3` convolutions.
    groups : int
        Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
        bottleneck blocks.
    norm_act : callable
        Function to create normalization / activation Module.
    dropout: callable
        Function to create Dropout Module.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 conf,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_act=ABN,
                 dropout=None,
                 bilinearmodule=InterChannelUpDown):
        super(ResidualBlock_bilinear, self).__init__()

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]
        self.is_bottleneck = is_bottleneck
        bilinear_block = bilinearmodule
        self.need_index = stride==2
        self.downsample_group = conf.MODEL.downupsample_group
        self.upsample_kernelsize = conf.MODEL.up_kernel_size
        self.downsample_kernelsize = 2*self.upsample_kernelsize
        self.share = conf.MODEL.share

        if not is_bottleneck:
            bn2 = norm_act(channels[1])
            bn2.activation = "identity"
            self.layer1 = nn.Sequential(OrderedDict(
                [
                    ("conv1", nn.Conv2d(in_channels, channels[0], 3, stride=1, padding=dilation, bias=False,
                                        dilation=dilation)),
                    ("bn1", norm_act(channels[0])),
                ]
            )
            )
            if dropout is not None:
                self.layer2 = nn.Sequential(OrderedDict(
                    [
                        ("dropout", dropout()),
                        ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,
                                            dilation=dilation)),
                        ("bn2", bn2)
                    ]
                )
                )
            else:
                self.layer2 = nn.Sequential(OrderedDict(
                    [
                        ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,
                                            dilation=dilation)),
                        ("bn2", bn2)
                    ]
                )
                )
        else:
            bn3 = norm_act(channels[2])
            bn3.activation = "identity"
            self.layer1 = nn.Sequential(OrderedDict(
                [
                    ("conv1", nn.Conv2d(in_channels, channels[0], 3, stride=1, padding=dilation, bias=False,
                                        dilation=dilation)),
                    ("bn1", norm_act(channels[0])),
                ]
            )
            )
            self.layer2 = nn.Sequential(OrderedDict(
                [
                    ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,
                                        dilation=dilation)),
                    ("bn2", norm_act(channels[1]))
                ]
            )
            )
            if dropout is not None:
                self.layer3 = nn.Sequential(OrderedDict(
                    [
                        ("dropout", dropout()),
                        ("conv2", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False)),
                        ("bn2", bn3)
                    ]
                )
                )
            else:
                self.layer3 = nn.Sequential(OrderedDict(
                    [
                        ("conv2", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False)),
                        ("bn2", bn3)
                    ]
                )
                )
        if self.need_index:
            self.bilinear = bilinear_block(channels[0], norm_act, conf)

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)
            self.proj_bn = norm_act(channels[-1])
            self.proj_bn.activation = "identity"

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            residual = self.proj_conv(x)
            residual = self.proj_bn(residual)
        else:
            residual = x
        if not self.is_bottleneck:
            x0 = self.layer1(x)
            if self.need_index:
                idx_en, idx_de = self.bilinear(x0)
                x1 = InterChannelOperation.interchannel_downsampling(x0, idx_en, ik=self.downsample_kernelsize, gp=self.downsample_group, ratio=2)
            else:
                x1 = x0
            x = self.layer2(x1)
        else:
            x = self.layer1(x)
            x0 = self.layer2(x)
            if self.need_index:
                idx_en, idx_de = self.bilinear(x0)
                x1 = InterChannelOperation.interchannel_downsampling(x0, idx_en, ik=self.downsample_kernelsize, gp=self.downsample_group, ratio=2)
            else:
                x1 = x0
            x = self.layer3(x1)

        x = x + residual

        if self.layer1.bn1.activation == "leaky_relu":
            if self.need_index:
                return functional.leaky_relu(x, negative_slope=self.layer1.bn1.activation_param, inplace=True), idx_de, x0
            else:
                return functional.leaky_relu(x, negative_slope=self.layer1.bn1.activation_param, inplace=True)
        elif self.layer1.bn1.activation == "elu":
            if self.need_index:
                return functional.elu(x, alpha=self.layer1.bn1.activation_param, inplace=True), idx_de, x0
            else:
                return functional.elu(x, alpha=self.layer1.bn1.activation_param, inplace=True)
        elif self.layer1.bn1.activation == "identity":
            if self.need_index:
                return x, idx_de, x0
            else:
                return x



def pred(inp, oup, conv_operator, k, batch_norm):
    # the last 1x1 convolutional layer is very important
    hlConv2d = hlconv[conv_operator]
    return nn.Sequential(
        hlConv2d(inp, oup, k, 1, batch_norm),
        nn.Conv2d(oup, oup, k, 1, padding=k//2, bias=False)
    )



class ResNet34DecoderLearning(nn.Module):
    def __init__(
            self,
            conf,
            distribute,
            structure,
            bottleneck,
            dilation=1,
    ):
        super(ResNet34DecoderLearning, self).__init__()
        self.conf = conf
        self.output_stride = conf.MODEL.stride
        self.structure = structure
        self.bottleneck = bottleneck
        self.apply_aspp = conf.MODEL.aspp
        self.decoder_block_num = conf.MODEL.decoder_block_num
        self.decoder_conv_operator = conf.MODEL.decoder_conv_operator
        self.decoder_kernel_size = conf.MODEL.decoder_kernel_size
        self.distribute = distribute
        assert conf.TRAIN.crop_size % self.output_stride == 0


        if len(structure) != 4:
            raise ValueError("Expected a structure with four values")
        if dilation != 1 and len(dilation) != 4:
            raise ValueError("If dilation is not 1 it must contain four values")

        aspp = ASPP
        norm_act = InPlaceABNSync if distribute else InPlaceABN
        self.group = conf.MODEL.downupsample_group
        self.up_kernel_size = conf.MODEL.up_kernel_size
        self.down_kernel_size = 2 * conf.MODEL.up_kernel_size
        decoder_block = BilinearDecoder
        bilinear_block = InterChannelUpDown


        ### encoder ###
        self.layer0 = nn.Sequential(
            nn.Conv2d(4, 64, 7, stride=1, padding=3, bias=False),
            norm_act(64),
        )
        # downsample1 & downsample2
        self.bilinear0 = bilinear_block(64, norm_act, conf)
        self.bilinear1 = bilinear_block(64, norm_act, conf)

        in_channels = 64
        if self.bottleneck:
            channels = (64, 64, 256)
        else:
            channels = (64, 64)
        # layer1
        blocks = []
        for block_id in range(structure[0]):
            blocks.append(("block%d" % (block_id + 1), ResidualBlock(in_channels, channels, norm_act=norm_act, stride=1, dilation=try_index(dilation, 0))))
            # Update channels and p_keep
            in_channels = channels[-1]
        self.layer1 = nn.Sequential(OrderedDict(blocks))
        channels = [c * 2 for c in channels]
        # layer2
        self.layer2_1 = ResidualBlock_bilinear(in_channels, channels, conf, norm_act=norm_act, stride=2, dilation=try_index(dilation, 0), bilinearmodule=bilinear_block)

        in_channels = channels[-1]
        blocks = []
        for block_id in range(structure[1]-1):
            blocks.append(("block%d" % (block_id + 1), ResidualBlock(in_channels, channels, norm_act=norm_act, stride=1, dilation=try_index(dilation, 1))))
            # Update channels and p_keep
            in_channels = channels[-1]
        self.layer2_2 = nn.Sequential(OrderedDict(blocks))

        channels = [c * 2 for c in channels]
        # layer3
        self.layer3_1 = ResidualBlock_bilinear(in_channels, channels, conf, norm_act=norm_act, stride=2, dilation=try_index(dilation, 1), bilinearmodule=bilinear_block)

        in_channels = channels[-1]
        blocks = []
        for block_id in range(structure[2] - 1):
            blocks.append(("block%d" % (block_id + 1),
                ResidualBlock(in_channels, channels, norm_act=norm_act, stride=1, dilation=try_index(dilation, 2))))
            # Update channels and p_keep
            in_channels = channels[-1]
        self.layer3_2 = nn.Sequential(OrderedDict(blocks))


        # freeze encoder batch norm layers
        if conf.TRAIN.freeze_bn:
            self.freeze_bn()

        ### context aggregation ###
        if self.apply_aspp:
            self.dconv_pp = aspp(256, 256, output_stride=self.output_stride, batch_norm=norm_act)

        ### decoder ###
        self.decoder_layer4 = decoder_block(256, 128, conv_operator=self.decoder_conv_operator, kernel_size=self.decoder_kernel_size,
                                            batch_norm=norm_act)
        self.decoder_layer3 = decoder_block(128, 64, conv_operator=self.decoder_conv_operator, kernel_size=self.decoder_kernel_size,
                                            batch_norm=norm_act)
        self.decoder_layer2 = decoder_block(64, 64, conv_operator=self.decoder_conv_operator, kernel_size=self.decoder_kernel_size,
                                            batch_norm=norm_act)
        self.decoder_layer1 = decoder_block(64, 32, conv_operator=self.decoder_conv_operator, kernel_size=self.decoder_kernel_size,
                                            batch_norm=norm_act)
        self.short3 = self.make_short_layer(128, 256, norm_act)
        self.short2 = self.make_short_layer(64, 128, norm_act)
        self.short1 = self.make_short_layer(64, 64, norm_act)
        self.short0 = self.make_short_layer(4, 64, norm_act)

        self.pred = pred(32, 1, 'std_conv', k=self.decoder_kernel_size, batch_norm=norm_act)

        self._initialize_weights(conf)


    def forward(self, x):
        # encode
        l0 = self.layer0(x)  # 4x320x320

        idx0_en, idx0_de = self.bilinear0(l0)
        l0p = InterChannelOperation.interchannel_downsampling(l0, idx0_en, self.down_kernel_size, self.group)  # 64x160x160

        idx1_en, idx1_de = self.bilinear1(l0p)
        l0_1p = InterChannelOperation.interchannel_downsampling(l0p, idx1_en, self.down_kernel_size, self.group)  # 64x80x80


        l1 = self.layer1(l0_1p)  # 64x80x80
        l2p, idx2_de, l10 = self.layer2_1(l1)
        l2 = self.layer2_2(l2p)

        l3p, idx3_de, l20 = self.layer3_1(l2)  # 32x80x80
        l3 = self.layer3_2(l3p)


        # pyramid pooling
        if self.apply_aspp:
            l_m = self.dconv_pp(l3)  # 160x10x10
        else:
            l_m = l3

        # decode
        lcat_3 = self.short3(l2)
        l = self.decoder_layer4(l_m, lcat_3, idx_up=idx3_de, ik=self.up_kernel_size, gp=self.group)
        lcat_2 = self.short2(l1)
        l = self.decoder_layer3(l, lcat_2, idx_up=idx2_de, ik=self.up_kernel_size, gp=self.group)
        lcat_1 = self.short1(l0p)
        l = self.decoder_layer2(l, lcat_1, idx_up=idx1_de, ik=self.up_kernel_size, gp=self.group)
        lcat_0 = self.short0(x)
        l = self.decoder_layer1(l, lcat_0, idx_up=idx0_de, ik=self.up_kernel_size, gp=self.group)

        l_out = self.pred(l)

        return l_out

    def make_short_layer(self, inchannel, outchannel, norm_act):
        return nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            norm_act(outchannel),
        )

    def freeze_bn(self):
        for name, m in self.named_modules():
            if 'index' in name or 'bilinear' in name:
                continue
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()
            elif isinstance(m, ABN):
                m.eval()


    def _initialize_weights(self, conf):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                init_fn = getattr(nn.init, conf.MODEL.weight_init + '_')
                if conf.MODEL.weight_init.startswith("xavier") or conf.MODEL.weight_init == "orthogonal":
                    gain = conf.MODEL.weight_gain_multiplier
                    if conf.MODEL.activation == "relu" or conf.MODEL.activation == "elu":
                        gain *= nn.init.calculate_gain("relu")
                    elif conf.MODEL.activation == "leaky_relu":
                        gain *= nn.init.calculate_gain("leaky_relu", conf.MODEL.activation_param)
                    init_fn(m.weight, gain)
                elif conf.MODEL.weight_init.startswith("kaiming"):
                    if conf.MODEL.activation == "relu" or conf.MODEL.activation == "elu":
                        init_fn(m.weight, 0)
                    else:
                        init_fn(m.weight, conf.MODEL.activation_param)

                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, ABN):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, .1)
                nn.init.constant_(m.bias, 0.)



def resnetmat(pretrained=False, encoder='resnet34', **kwargs):
    """Constructs a MobileNet_V2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    _NETS = {
        "34": {"structure": [3, 4, 4, 2], "bottleneck": False},
    }

    if encoder == 'resnet34':
        model = ResNet34DecoderLearning(**kwargs, structure=_NETS["34"]["structure"], bottleneck=_NETS["34"]["bottleneck"])
    else:
        raise NotImplementedError

    if pretrained:
        corresp_name = CORRESP_NAME
        model_dict = model.state_dict()
        pretrained_dict = load_url(model_urls['resnet34'])
        pretrained_dict = pretrained_dict['state_dict']
        for name in pretrained_dict:
            if name not in corresp_name:
                continue
            if corresp_name[name] not in model_dict.keys():
                continue
            if name == "module.mod1.conv1.weight":
                model_weight = model_dict[corresp_name[name]]
                assert model_weight.shape[1] == 4
                model_weight[:, 0:3, :, :] = pretrained_dict[name]
                model_weight[:, 3, :, :] = torch.tensor(0)
                model_dict[corresp_name[name]] = model_weight
            else:
                model_dict[corresp_name[name]] = pretrained_dict[name]
        model.load_state_dict(model_dict)

    return model

def load_url(url, model_dir='./pretrained', model_name='resnet34.pth.tar', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = model_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)
