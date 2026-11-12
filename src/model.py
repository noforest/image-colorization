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
        
        self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, out_channels=64, kernel_size=3, padding=1), # (B, 4, 256, 256) -> (B, 64, 256, 256)
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (B, 64, 256, 256) 
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2), # (B, 64, 128, 128)

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # (B, 128, 128, 128)
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (B, 128, 128, 128)
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2), # (B, 128, 64, 64)

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # (B, 256, 64, 64)
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # (B, 256, 64, 64)
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2), # (B, 256, 32, 32)

                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1), # (B, 512, 32, 32)
                nn.BatchNorm2d(512),
                nn.ReLU(),
        )

        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256)
        )



