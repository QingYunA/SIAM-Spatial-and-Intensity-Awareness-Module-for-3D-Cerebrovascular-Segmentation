from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

# from torchsummary import summary


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


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=64):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(UNetEncoder, self).__init__()

        features = init_features
        self.encoder1 = UnetBlock(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UnetBlock(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UnetBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UnetBlock(features * 4, features * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UnetBlock(features * 8, features * 16)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        return bottleneck, [enc1, enc2, enc3, enc4]


class UNetDecoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=64, SIAM=False):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """
        features = init_features
        super().__init__()
        if SIAM:
            self.SIAM = 2
        else:
            self.SIAM = 1
        self.upconv4 = nn.ConvTranspose3d(features * 16 * self.SIAM, features * 8 * self.SIAM, kernel_size=2, stride=2)
        self.decoder4 = UnetBlock((features * 8) * 2 * self.SIAM, features * 8 * self.SIAM)
        self.upconv3 = nn.ConvTranspose3d(features * 8 * self.SIAM, features * 4 * self.SIAM, kernel_size=2, stride=2)
        self.decoder3 = UnetBlock((features * 4) * 2 * self.SIAM, features * 4 * self.SIAM)
        self.upconv2 = nn.ConvTranspose3d(features * 4 * self.SIAM, features * 2 * self.SIAM, kernel_size=2, stride=2)
        self.decoder2 = UnetBlock((features * 2) * 2 * self.SIAM, features * 2 * self.SIAM)
        self.upconv1 = nn.ConvTranspose3d(features * 2 * self.SIAM, features * self.SIAM, kernel_size=2, stride=2)
        self.decoder1 = UnetBlock(features * 2 * self.SIAM, features * self.SIAM)

        self.conv = nn.Conv3d(in_channels=features * self.SIAM, out_channels=out_channels, kernel_size=1)

    def forward(self, encoder_x, encs):
        enc1, enc2, enc3, enc4 = encs
        dec4 = self.upconv4(encoder_x)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        outputs = self.conv(dec1)

        return outputs


class SIAMUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, module_list=["IAM", "SAM"]) -> None:
        super().__init__()
        features = init_features
        self.module_list = module_list
        # backbone
        self.unet_encoder = UNetEncoder(in_channels, out_channels, init_features=features)
        self.unet_decoder = UNetDecoder(in_channels, out_channels, init_features=features, SIAM=False)
        if "SAM" in self.module_list:
            self.spatial_decoder = UNetDecoder(in_channels=in_channels, out_channels=1, init_features=features, SIAM=False)
            self.convx1 = UnetBlock(in_planes=features * 16, out_planes=features * 16)
        if "IAM" in self.module_list:
            self.intensity_decoder = UNetDecoder(in_channels=in_channels, out_channels=1, init_features=features, SIAM=False)
            self.convx2 = UnetBlock(in_planes=features * 16, out_planes=features * 16)
        self.convx3 = UnetBlock(in_planes=features * 16, out_planes=features * 16)

    def forward(self, source_x, spatial_x, intensity_x):
        out, encs = self.unet_encoder(source_x)
        spatial_d_out = 0
        intensity_d_out = 0

        if self.training:
            if "SAM" in self.module_list:
                with torch.no_grad():
                    spatial_out, spatial_encs = self.unet_encoder(spatial_x)
                spatial_out = self.convx1(spatial_out)
                # with torch.no_grad():
                #     out = self.convx1(out)
                spatial_d_out = self.spatial_decoder(spatial_out, spatial_encs)
            if "IAM" in self.module_list:
                with torch.no_grad():
                    intensity_out, intensity_encs = self.unet_encoder(intensity_x)
                intensity_out = self.convx2(intensity_out)
                # with torch.no_grad():
                #     out = self.convx2(out)
                intensity_d_out = self.intensity_decoder(intensity_out, intensity_encs)

            out = self.convx3(out)
            out = self.unet_decoder(out, encs)

            return out, spatial_d_out, intensity_d_out
        else:
            out = self.convx3(out)
            out = self.unet_decoder(out, encs)
            return out


if __name__ == "__main__":
    import torch.nn as nn

    x = torch.randn(4, 3, 64, 64, 64)
    x_1 = torch.randn(4, 1, 64, 64, 64)
    model = SIAMUNet(in_channels=1, out_channels=2, init_features=32)
    model.train()
    y, s_y, i_y = model(x)
    mse_loss = nn.MSELoss()
    # print(y.size())
    # print(i_y)
    loss = mse_loss(x_1, i_y)
    print(x_1)
    print(i_y)
    print(loss)
