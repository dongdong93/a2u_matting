"""
This implementation is modified from the following repository:
https://github.com/jfzhang95/pytorch-deeplab-xception
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inplace_abn import ABN


def depth_sep_dilated_conv_3x3_bn(inp, oup, padding, dilation, BatchNorm2d=ABN):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, 1, padding=padding, dilation=dilation, groups=inp, bias=False),
        BatchNorm2d(inp),
        nn.Conv2d(inp, oup, 1, 1, padding=0, bias=False),
        BatchNorm2d(oup)
    )

def dilated_conv_3x3_bn(inp, oup, padding, dilation, BatchNorm2d=ABN):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, 1, padding=padding, dilation=dilation, bias=False),
        BatchNorm2d(oup)
    )

class _ASPPModule(nn.Module):
    def __init__(self, inp, planes, kernel_size, padding, dilation, batch_norm):
        super(_ASPPModule, self).__init__()
        BatchNorm2d = batch_norm
        if kernel_size == 1:
            self.atrous_conv = nn.Sequential(
                nn.Conv2d(inp, planes, kernel_size=1, stride=1, padding=padding, dilation=dilation, bias=False),
                BatchNorm2d(planes)
            )
        elif kernel_size == 3:
            # we use depth-wise separable convolution to save the number of parameters
            self.atrous_conv = depth_sep_dilated_conv_3x3_bn(inp, planes, padding, dilation, BatchNorm2d)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, ABN):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inp, oup, output_stride=32, batch_norm=ABN):
        super(ASPP, self).__init__()

        if output_stride == 32:
            dilations = [1, 2, 4, 8]
        elif output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        BatchNorm2d = batch_norm

        self.aspp1 = _ASPPModule(inp, 256, 1, padding=0, dilation=dilations[0], batch_norm=BatchNorm2d)
        self.aspp2 = _ASPPModule(inp, 256, 3, padding=dilations[1], dilation=dilations[1], batch_norm=BatchNorm2d)
        self.aspp3 = _ASPPModule(inp, 256, 3, padding=dilations[2], dilation=dilations[2], batch_norm=BatchNorm2d)
        self.aspp4 = _ASPPModule(inp, 256, 3, padding=dilations[3], dilation=dilations[3], batch_norm=BatchNorm2d)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inp, 256, 1, stride=1, padding=0, bias=False),
            BatchNorm2d(256)
        )

        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(256*5, oup, 1, stride=1, padding=0, bias=False),
            BatchNorm2d(oup)
        )
        
        self.dropout = nn.Dropout(0.5)

        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='nearest')
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.bottleneck_conv(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, ABN):
                m.weight.data.fill_(1)
                m.bias.data.zero_()