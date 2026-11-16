from model import UNet
import torch


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(in_channels=4).to(device)
    model.fit()
