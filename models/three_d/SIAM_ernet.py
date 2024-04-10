#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author : hao zhang
# @File   : model.py

# based on github project https://github.com/iMED-Lab/ER-Net
import torch
import torch.nn as nn
from functools import partial

import torch.nn.functional as F

nonlinearity = partial(F.relu, inplace=True)


def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResDecoder(nn.Module):
    def __init__(self, in_channels):
        super(ResDecoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out + residual
        out = self.relu(out)
        return out


class SFConv(nn.Module):
    def __init__(self, features, M=2, r=4, L=32):
        """Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SFConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        # self.convs = nn.ModuleList([])
        # for i in range(M):
        #     self.convs.append(nn.Sequential(
        #         nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
        #         nn.BatchNorm2d(features),
        #         nn.ReLU(inplace=False)
        #     ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        # for i, conv in enumerate(self.convs):
        #     fea = conv(x).unsqueeze_(dim=1)
        #     if i == 0:
        #         feas = fea
        #     else:
        #         feas = torch.cat([feas, fea], dim=1)
        feas = torch.cat((x1.unsqueeze_(dim=1), x2.unsqueeze_(dim=1)), dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1).mean((-1))
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class SF_Decoder(nn.Module):
    def __init__(self, out_channels):
        super(SF_Decoder, self).__init__()
        self.conv1 = SFConv(out_channels)
        self.bn1 = nn.BatchNorm3d(out_channels)
        # self.conv2 = nn.Conv3d(out_channels, out_channels // 2, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm3d(out_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.ResDecoder = ResDecoder(out_channels)
        # self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        # residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x1, x2)))
        out = self.ResDecoder(out)

        # out = self.relu(self.bn2(self.conv2(out)))
        # out += residual
        # out = self.relu(out)
        return out


class ResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out + residual
        out = self.relu(out)
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
    def __init__(self, channels):
        super().__init__()
        self.encoder1 = ResEncoder(channels, 32)
        self.encoder2 = ResEncoder(32, 64)
        self.encoder3 = ResEncoder(64, 128)

        self.down = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bridge = ResEncoder(128, 256)

        self.conv1_1 = nn.Conv3d(256, 1, kernel_size=1)
        self.conv2_2 = nn.Conv3d(128, 1, kernel_size=1)
        self.conv3_3 = nn.Conv3d(64, 1, kernel_size=1)

        self.convTrans1 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        self.convTrans2 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        self.convTrans3 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        down1 = self.down(enc1)

        enc2 = self.encoder2(down1)
        down2 = self.down(enc2)

        con3_3 = self.conv3_3(enc2)
        convTrans3 = self.convTrans3(con3_3)
        x3 = -1 * (torch.sigmoid(convTrans3)) + 1
        x3 = x3.expand(-1, 32, -1, -1, -1).mul(enc1)
        x3 = x3 + enc1

        enc3 = self.encoder3(down2)
        down3 = self.down(enc3)

        con2_2 = self.conv2_2(enc3)
        convTrans2 = self.convTrans2(con2_2)
        x2 = -1 * (torch.sigmoid(convTrans2)) + 1
        x2 = x2.expand(-1, 64, -1, -1, -1).mul(enc2)
        x2 = x2 + enc2

        bridge = self.bridge(down3)

        conv1_1 = self.conv1_1(bridge)
        convTrans1 = self.convTrans1(conv1_1)

        x = -1 * (torch.sigmoid(convTrans1)) + 1
        x = x.expand(-1, 128, -1, -1, -1).mul(enc3)
        x = x + enc3

        return bridge, [x, x2, x3]


class Decoder(nn.Module):
    def __init__(self, classes) -> None:
        super().__init__()
        self.decoder3 = SF_Decoder(128)
        self.decoder2 = SF_Decoder(64)
        self.decoder1 = SF_Decoder(32)
        self.down = downsample()
        self.up3 = deconv(256, 128)
        self.up2 = deconv(128, 64)
        self.up1 = deconv(64, 32)
        self.final = nn.Conv3d(32, classes, kernel_size=1, padding=0)

    def forward(self, bridge, x_layer):
        x, x2, x3 = x_layer
        up3 = self.up3(bridge)
        # up3 = SKII_Decoder(up3,)

        # up3 = torch.cat((up3, x), dim=1)
        dec3 = self.decoder3(up3, x)

        up2 = self.up2(dec3)
        # up2 = torch.cat((up2, x2), dim=1)
        dec2 = self.decoder2(up2, x2)

        up1 = self.up1(dec2)
        # up1 = torch.cat((up1, x3), dim=1)
        dec1 = self.decoder1(up1, x3)

        final = self.final(dec1)
        return final


class SIAM_ER_Net(nn.Module):
    def __init__(self, classes, channels):
        super().__init__()
        self.encoder = Encoder(channels=channels)
        self.decoder = Decoder(classes=classes)

        self.s_decoder = Decoder(classes=classes)
        self.i_decoder = Decoder(classes=classes)
        self.convx1 = UnetBlock(in_planes=256, out_planes=256)
        self.convx2 = UnetBlock(in_planes=256, out_planes=256)
        self.convx3 = UnetBlock(in_planes=256, out_planes=256)

    def forward(self, x, s_x, i_x):
        out, encs = self.encoder(x)
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


class ER_Net(nn.Module):
    def __init__(self, classes, channels):
        # def __init__(self):

        super(ER_Net, self).__init__()
        self.encoder1 = ResEncoder(channels, 32)
        self.encoder2 = ResEncoder(32, 64)
        self.encoder3 = ResEncoder(64, 128)
        self.bridge = ResEncoder(128, 256)

        self.conv1_1 = nn.Conv3d(256, 1, kernel_size=1)
        self.conv2_2 = nn.Conv3d(128, 1, kernel_size=1)
        self.conv3_3 = nn.Conv3d(64, 1, kernel_size=1)

        self.convTrans1 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        self.convTrans2 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        self.convTrans3 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)

        self.decoder3 = SF_Decoder(128)
        self.decoder2 = SF_Decoder(64)
        self.decoder1 = SF_Decoder(32)
        self.down = downsample()
        self.up3 = deconv(256, 128)
        self.up2 = deconv(128, 64)
        self.up1 = deconv(64, 32)
        self.final = nn.Conv3d(32, classes, kernel_size=1, padding=0)
        # initialize_weights(self)

    def forward(self, x):
        enc1 = self.encoder1(x)
        down1 = self.down(enc1)

        enc2 = self.encoder2(down1)
        down2 = self.down(enc2)

        con3_3 = self.conv3_3(enc2)
        convTrans3 = self.convTrans3(con3_3)
        x3 = -1 * (torch.sigmoid(convTrans3)) + 1
        x3 = x3.expand(-1, 32, -1, -1, -1).mul(enc1)
        x3 = x3 + enc1

        enc3 = self.encoder3(down2)
        down3 = self.down(enc3)

        con2_2 = self.conv2_2(enc3)
        convTrans2 = self.convTrans2(con2_2)
        x2 = -1 * (torch.sigmoid(convTrans2)) + 1
        x2 = x2.expand(-1, 64, -1, -1, -1).mul(enc2)
        x2 = x2 + enc2

        bridge = self.bridge(down3)

        conv1_1 = self.conv1_1(bridge)
        convTrans1 = self.convTrans1(conv1_1)

        x = -1 * (torch.sigmoid(convTrans1)) + 1
        x = x.expand(-1, 128, -1, -1, -1).mul(enc3)
        x = x + enc3

        up3 = self.up3(bridge)
        # up3 = SKII_Decoder(up3,)

        # up3 = torch.cat((up3, x), dim=1)
        dec3 = self.decoder3(up3, x)

        up2 = self.up2(dec3)
        # up2 = torch.cat((up2, x2), dim=1)
        dec2 = self.decoder2(up2, x2)

        up1 = self.up1(dec2)
        # up1 = torch.cat((up1, x3), dim=1)
        dec1 = self.decoder1(up1, x3)

        final = self.final(dec1)
        # final = F.sigmoid(final)
        return final
