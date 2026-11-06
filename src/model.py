import torch
import torch.nn as nn


class UNet(nn.Module):
    """ 
    - Image d'entrée doit être 256 par 256
    - idée: https://miro.medium.com/v2/resize:fit:720/format:webp/1*1wV5-9EVb_3KmNjfuoTqkw.png
    """
    def __init__(self):
        
        self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1), # (B, 1, 256, 256) -> (B, 64, 256, 256)
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (B, 1, 254, 254) -> (B, 64, 252, 252)
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2), # (B, 64, 125, 125)

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # (B, 128, 123, 123)
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (B, 128, 121, 121)
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2), # (B, 128, ?, ?)

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
        )

        self.decoder = nn.Sequential(
                # ?
                )



