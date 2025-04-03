import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, base_c=64):
        super().__init__()
        self.dc1 = DoubleConv(in_channels, base_c)
        self.pool = nn.MaxPool2d(2, 2)
        self.dc2 = DoubleConv(base_c, base_c*2)
        self.dc3 = DoubleConv(base_c*2, base_c*4)
        self.dc4 = DoubleConv(base_c*4, base_c*8)
        self.dc5 = DoubleConv(base_c*8, base_c*16)

        self.up4 = nn.ConvTranspose2d(base_c*16, base_c*8, kernel_size=2, stride=2)
        self.uc4 = DoubleConv(base_c*16, base_c*8)
        self.up3 = nn.ConvTranspose2d(base_c*8, base_c*4, kernel_size=2, stride=2)
        self.uc3 = DoubleConv(base_c*8, base_c*4)
        self.up2 = nn.ConvTranspose2d(base_c*4, base_c*2, kernel_size=2, stride=2)
        self.uc2 = DoubleConv(base_c*4, base_c*2)
        self.up1 = nn.ConvTranspose2d(base_c*2, base_c, kernel_size=2, stride=2)
        self.uc1 = DoubleConv(base_c*2, base_c)

        self.out_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.dc1(x)       
        p1 = self.pool(c1)

        c2 = self.dc2(p1)
        p2 = self.pool(c2)

        c3 = self.dc3(p2)
        p3 = self.pool(c3)

        c4 = self.dc4(p3)
        p4 = self.pool(c4)

        c5 = self.dc5(p4)

        u4 = self.up4(c5)
        cat4 = torch.cat([u4, c4], dim=1)
        c6 = self.uc4(cat4)

        u3 = self.up3(c6)
        cat3 = torch.cat([u3, c3], dim=1)
        c7 = self.uc3(cat3)

        u2 = self.up2(c7)
        cat2 = torch.cat([u2, c2], dim=1)
        c8 = self.uc2(cat2)

        u1 = self.up1(c8)
        cat1 = torch.cat([u1, c1], dim=1)
        c9 = self.uc1(cat1)

        out = self.out_conv(c9)
        return out