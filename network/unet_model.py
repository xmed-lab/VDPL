""" Full assembly of the parts to form the complete network """

from network.unet_parts import *
from network.mixstyle import MixStyle

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, norm_layer=nn.BatchNorm2d):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, norm_layer=norm_layer)
        self.down1 = Down(64, 128, norm_layer=norm_layer)
        self.down2 = Down(128, 256, norm_layer=norm_layer)
        self.down3 = Down(256, 512, norm_layer=norm_layer)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, norm_layer=norm_layer)
        self.up1 = Up(1024, 512 // factor, bilinear, norm_layer=norm_layer)
        self.up2 = Up(512, 256 // factor, bilinear, norm_layer=norm_layer)
        self.up3 = Up(256, 128 // factor, bilinear, norm_layer=norm_layer)
        self.up4 = Up(128, 64, bilinear, norm_layer=norm_layer)
        self.outc = OutConv(64, n_classes)

        self.mixstyle = MixStyle(p=0.5, alpha=0.1)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.mixstyle(x1)
        x2 = self.down1(x1)
        x2 = self.mixstyle(x2)
        x3 = self.down2(x2)
        x3 = self.mixstyle(x3)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

