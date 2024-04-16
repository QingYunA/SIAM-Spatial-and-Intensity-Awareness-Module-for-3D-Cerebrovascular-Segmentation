# 3D-UNet model.
# x: 128x128 resolution for 32 frames.
# https://github.com/huangzhii/FCN-3D-pytorch/blob/master/main3d.py
import torch
import torch.nn as nn
import os
import numpy as np
from collections import OrderedDict


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, in_channels, elu):
        super(InputTransition, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(self.num_features)

        self.relu1 = ELUCons(elu, self.num_features)

    def forward(self, x):
        out = self.conv1(x)
        repeat_rate = int(self.num_features / self.in_channels)
        out = self.bn1(out)
        x16 = x.repeat(1, repeat_rate, 1, 1, 1)
        return self.relu1(torch.add(out, x16))


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm3d(outChans)

        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)

        self.bn1 = torch.nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, in_channels, classes, elu):
        super(OutputTransition, self).__init__()
        self.classes = classes
        self.conv1 = nn.Conv3d(in_channels, classes, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(classes)

        self.conv2 = nn.Conv3d(classes, classes, kernel_size=1)
        self.relu1 = ELUCons(elu, classes)

    def forward(self, x):
        # convolve 32 down to channels as the desired classes
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out


class UnetBlock(nn.Module):
    def __init__(self, in_planes, out_planes) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class Encoder(nn.Module):
    def __init__(self, elu=True, in_channels=1, classes=1):
        super().__init__()
        self.classes = classes
        self.in_channels = in_channels

        self.in_tr = InputTransition(in_channels, elu=elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=False)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=False)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)

        return out256, [out128, out64, out32, out16]


class Decoder(nn.Module):
    def __init__(self, elu=True, in_channels=1, classes=1):
        super().__init__()
        self.classes = classes
        self.in_channels = in_channels

        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=False)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=False)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, classes, elu)

    def forward(self, x, skips):
        out = self.up_tr256(x, skips[0])
        out = self.up_tr128(out, skips[1])
        out = self.up_tr64(out, skips[2])
        out = self.up_tr32(out, skips[3])
        out = self.out_tr(out)
        return out


class SIAMVNet(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, classes=classes)
        self.decoder = Decoder(in_channels=in_channels, classes=classes)

        self.s_decoder = Decoder(in_channels=in_channels, classes=classes)
        self.i_decoder = Decoder(in_channels=in_channels, classes=classes)
        self.convx1 = UnetBlock(in_planes=256, out_planes=256)
        self.convx2 = UnetBlock(in_planes=256, out_planes=256)
        self.convx3 = UnetBlock(in_planes=256, out_planes=256)

    def forward(self, x, s_x, i_x):
        out, encs = self.encoder(x)
        if self.training:
            with torch.no_grad():
                s_out, s_encs = self.encoder(s_x)
            s_out = self.convx1(s_out)
            s_out = self.s_decoder(s_out, s_encs)

            with torch.no_grad():
                i_out, i_encs = self.encoder(i_x)
            i_out = self.convx2(i_out)
            i_out = self.i_decoder(i_out, i_encs)

            out = self.convx3(out)
            out = self.decoder(out, encs)

            return out, s_out, i_out
        else:
            out = self.convx3(out)
            out = self.decoder(out, encs)
            return out
