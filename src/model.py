from unet_pieces import UpConv
from unet_pieces import DownConv
from unet_pieces import DoubleConv
import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    - Image d'entrée doit être 256 par 256
    - idée: https://miro.medium.com/v2/resize:fit:720/format:webp/1*1wV5-9EVb_3KmNjfuoTqkw.png
    - sortie = (taille - kernel + 2*padding) / stride + 1
        ex pour MaxPool2d(2) : stride par défaut c'est 1

    - cf: https://github.com/uygarkurt/UNet-PyTorch/blob/main/unet.py
    [essayer de le refaire comme un grand ouaaaaaaai]
    """
    def __init__(self, in_channels=4):

        self.downconv1 = DownConv(in_channels, 64)
        self.downconv2 = DownConv(64, 128)
        self.downconv3 = DownConv(128, 256)

        self.bottle_neck = DoubleConv(256, 512)

        self.upconv3 = UpConv(512, 256)
        self.upconv2 = UpConv(256, 128)
        self.upconv1 = UpConv(128, 64)

        self.out = nn.Conv2d(64, out_channels=2, kernel_size=3) # nb de classes = 2 ==> couleurs a, b

    def forward(self, x):
        down1, p1 = self.downconv1(x)
        down2, p2 = self.downconv2(p1)
        down3, p3 = self.downconv3(p2)

        b = self.bottle_neck(p3)

        up3 = self.upconv3(b, down3)
        up2 = self.upconv2(up3, down2)
        up1 = self.upconv1(up2, down1)

        return self.out(up1)
