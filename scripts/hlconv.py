
import torch
import torch.nn as nn
import torch.nn.functional as functional
from collections import OrderedDict
from inplace_abn import ABN

def conv_bn(inp, oup, k=3, s=1, BatchNorm2d=ABN):
    return nn.Sequential(
        nn.Conv2d(inp, oup, k, s, padding=k//2, bias=False),
        BatchNorm2d(oup)
    )


def dep_sep_conv_bn(inp, oup, k=3, s=1, BatchNorm2d=ABN):
    return nn.Sequential(
        nn.Conv2d(inp, inp, k, s, padding=k//2, groups=inp, bias=False),
        BatchNorm2d(inp),
        nn.Conv2d(inp, oup, 1, 1, padding=0, bias=False),
        BatchNorm2d(oup)
    )

def appro_dep_sep_conv_bn(inp, oup, k=5, s=1, BatchNorm2d=ABN):
    return nn.Sequential(
        nn.Conv2d(inp, inp, [k, 1], s, padding=[k//2, 0], groups=inp, bias=False),
        BatchNorm2d(inp),
        nn.Conv2d(inp, inp, [1, k], s, padding=[0, k//2], groups=inp, bias=False),
        BatchNorm2d(inp),
        nn.Conv2d(inp, oup, 1, 1, padding=0, bias=False),
        BatchNorm2d(oup)
    )

def inverted_appro_dep_sep_conv_bn(inp, oup, k=5, s=1,  BatchNorm2d=ABN):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, padding=0, bias=False),
        BatchNorm2d(oup),
        nn.Conv2d(oup, oup, [k, 1], s, padding=[k//2, 0], groups=oup, bias=False),
        BatchNorm2d(oup),
        nn.Conv2d(oup, oup, [1, k], s, padding=[0, k//2], groups=oup, bias=False),
        BatchNorm2d(oup)
    )

class ApproDepSepResidualBlock(nn.Module):
    def __init__(self, inp, oup, k=5, s=1, BatchNorm2d=ABN):
        super(ApproDepSepResidualBlock, self).__init__()
        self.dep_sep_conv = nn.Sequential(
            nn.Conv2d(inp, inp, [k, 1], s, padding=[k//2, 0], groups=inp, bias=False),
            BatchNorm2d(inp),
            nn.Conv2d(inp, inp, [1, k], s, padding=[0, k//2], groups=inp, bias=False),
            BatchNorm2d(inp),
        )
        self.column_conv = conv_bn(inp, oup, 1, 1, BatchNorm2d)
        self._init_weight()

    def forward(self, x):
        return self.column_conv(x + self.dep_sep_conv(x))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ResidualBlock(nn.Module):
    def __init__(self, inp, oup, k=5, s=1, BatchNorm2d=ABN):
        super(ResidualBlock, self).__init__()
        bn2 = BatchNorm2d(oup)
        bn2.activation = "identity"
        self.dep_sep_conv =  nn.Sequential(OrderedDict(
                [
                    ("conv1", nn.Conv2d(inp, oup, k, s, padding=k//2, bias=False)),
                    ("bn1", BatchNorm2d(oup)),
                    ("conv2", nn.Conv2d(oup, oup, k, s, padding=k//2, bias=False)),
                    ("bn2", bn2),
                ]
            ))

        self.need_proj_conv = inp != oup
        if self.need_proj_conv:
            self.proj_conv = nn.Conv2d(inp, oup, 1, stride=s, padding=0, bias=False)
            self.proj_bn = BatchNorm2d(oup)
            self.proj_bn.activation = "identity"

        self._init_weight()

    def forward(self, x):
        if self.need_proj_conv:
            residual = self.proj_conv(x)
            residual = self.proj_bn(residual)
        else:
            residual = x

        x = self.dep_sep_conv(x) + residual

        if self.dep_sep_conv.bn1.activation == "leaky_relu":
            return functional.leaky_relu(x, negative_slope=self.dep_sep_conv.bn1.activation_param, inplace=True)
        elif self.dep_sep_conv.bn1.activation == "elu":
            return functional.elu(x, alpha=self.dep_sep_conv.bn1.activation_param, inplace=True)
        elif self.dep_sep_conv.bn1.activation == "identity":
            return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, ABN):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

hlconv = {
    'std_conv': conv_bn,
    'dep_sep_conv': dep_sep_conv_bn,
    'appro_dep_sep_conv': appro_dep_sep_conv_bn,
    'inverted_appro_dep_sep_conv': inverted_appro_dep_sep_conv_bn,
    'appro_dep_sep_res_conv': ApproDepSepResidualBlock,
    'residual_conv': ResidualBlock
}