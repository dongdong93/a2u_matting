import torch
import torch.nn as nn

from hlconv import hlconv
from inplace_abn import ABN
from BilinearModel import InterChannelOperation


class BilinearDecoder(nn.Module):
    def __init__(self, inp, oup, conv_operator='std_conv', kernel_size=5, batch_norm=ABN):
        super(BilinearDecoder, self).__init__()
        hlConv2d = hlconv[conv_operator]
        BatchNorm2d = batch_norm

        # inp, oup, kernel_size, stride, batch_norm
        self.dconv = hlConv2d(inp, oup, kernel_size, 1, BatchNorm2d)
        self.dconv1 = hlConv2d(oup, oup, kernel_size, 1, BatchNorm2d)

        self._init_weight()

    def forward(self, l_encode, l_low, idx_up=None, ik=5, gp=1, ratio=1):
        if idx_up is not None:
            l_encode = InterChannelOperation.interchannel_upsampling(l_encode, idx_up, ik, gp)
        l_cat = l_encode + l_low if l_low is not None else l_encode
        return self.dconv1(self.dconv(l_cat))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, ABN):
                m.weight.data.fill_(1)
                m.bias.data.zero_()





