import torch
import torch.nn as nn
import torch.nn.functional as F
from hlconv import hlconv
from inplace_abn import ABN, InPlaceABN
from biupdownsample import bidownsample_naive, biupsample_naive
import math


class InterChannelUpDown(nn.Module):
    '''
    s: upsample kernel size
    group: upsample kernel group
    share: False means U and V are channelwise. True means U and V are shared among channels
    '''
    def __init__(self, c, BatchNorm2d, conf):
        super(InterChannelUpDown, self).__init__()
        self.inchannel = c
        self.s = conf.MODEL.up_kernel_size
        self.k = conf.MODEL.encode_kernel_size
        self.share = conf.MODEL.share
        self.group = conf.MODEL.downupsample_group
        self.d = 1
        self.k_u = self.k
        self.padding = int((self.k - 1) / 2) if self.k % 2 == 0 else self.k // 2
        self.padding_u = int((self.k_u - 1) / 2) if self.k_u % 2 == 0 else self.k_u // 2
        self.bnu = nn.Sequential(
            nn.GroupNorm(num_channels=c, num_groups=32),
            nn.LeakyReLU()
        )
        self.bnv = nn.Sequential(
            nn.GroupNorm(num_channels=c, num_groups=32),
            nn.LeakyReLU()
        )
        self.bconv_UP = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c, self.d * (self.s * 2) ** 2 * self.group, 1, 1, padding=0, bias=True),
        )
        self.bconv_DP = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c, self.d * (self.s * 2) ** 2 * self.group, 1, 1, padding=0, bias=True),
        )
        self.bconv_U = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.k_u, self.k_u)),
            nn.Conv2d(c, self.d if self.share else self.d * c, 1, 1, padding=0, groups=1, bias=True),
        )
        self.bconv_V = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.k, self.k)),
            nn.Conv2d(c, self.d if self.share else self.d * c, 1, 1, padding=0, groups=1, bias=True),
        )

    def forward(self, x):
        n, c, h, w = x.size()
        p_u = self.bconv_UP(x).view(n, (self.s*2)**2*self.group, self.d, 1, 1)
        p_d = self.bconv_DP(x).view(n, (self.s * 2) ** 2 * self.group, self.d, 1, 1)
        u = self.bconv_U(x).view(n, -1, self.k_u, self.k_u)
        v = self.bconv_V(x).view(n, -1, self.k, self.k)

        out_u, out_d = [], []

        for i, (u1, v1, p_u1, p_d1) in enumerate(zip(u, v, p_u, p_d)):
            out1, out2 = [], []
            for j in range(self.d):
                if self.share:
                    out1.append(F.conv2d(x[i].unsqueeze(1), u1[j].view(1, 1, self.k_u, self.k_u), bias=None, stride=2, padding=self.padding_u).view(1, c, int(h / 2), int(w / 2)))  # n*c*h*w
                    out2.append(F.conv2d(x[i].unsqueeze(1), v1[j].view(1, 1, self.k, self.k), bias=None, stride=2, padding=self.padding).view(1, c, int(h / 2), int(w / 2)))  # n*c*h*w
                else:
                    out1.append(F.conv2d(x[i].unsqueeze(0), u1[j*c:(j+1)*c].view(c, 1, self.k_u, self.k_u), bias=None, stride=2, groups=c, padding=self.padding_u).view(1, c, int(h / 2), int(w / 2)))  # n*c*h*w
                    out2.append(F.conv2d(x[i].unsqueeze(0), v1[j*c:(j+1)*c].view(c, 1, self.k, self.k), bias=None, stride=2, groups=c, padding=self.padding).view(1, c, int(h / 2), int(w / 2)))  # n*c*h*w
            out1 = self.bnu(torch.cat(out1, 0)).view(c, self.d, int(h / 2), int(w / 2))
            out2 = self.bnv(torch.cat(out2, 0)).view(c, self.d, int(h / 2), int(w / 2))

            out_u.append(F.conv2d((out1*out2).sum(dim=0, keepdim=True)/math.sqrt(c), p_u1.view((self.s*2)**2*self.group, self.d, 1, 1), stride=1, padding=0))
            out_d.append(F.conv2d((out1*out2).sum(dim=0, keepdim=True)/math.sqrt(c), p_d1.view((self.s * 2) ** 2 * self.group, self.d, 1, 1), stride=1, padding=0))

        out_u = torch.cat(out_u, 0)
        out_d = torch.cat(out_d, 0)
        out_u = out_u.view(n, self.s ** 2 * 4 * self.group, int(h / 2), int(w / 2))
        out_u = F.pixel_shuffle(out_u, 2).view(n, self.s ** 2, self.group, h, w)
        out_u = F.softmax(torch.sigmoid(out_u), dim=1).view(n, -1, h, w) if self.s > 1 else torch.sigmoid(out_u).view(n, -1, h, w)
        out_d = out_d.view(n, self.s ** 2 * 4 * self.group, int(h / 2), int(w / 2)).view(n, self.s ** 2 * 4, self.group, int(h / 2), int(w / 2))
        out_d = F.softmax(out_d, dim=1).view(n, -1, int(h / 2), int(w / 2))

        return out_d, out_u


class InterChannelOperation:
    @staticmethod
    def interchannel_upsampling(x, idx_up, ik, gp=1, ratio=2):
        # default upsampling ratio = 2
        x = biupsample_naive(x, idx_up, ik, gp, ratio)
        return x

    @staticmethod
    def interchannel_downsampling(x, idx_down, ik, gp=1, ratio=2):
        # default downsampling ratio = 2
        x = bidownsample_naive(x, idx_down, ik, gp, ratio)
        return x
