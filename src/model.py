import copy
from unet_pieces import UpConv
from unet_pieces import DownConv
from unet_pieces import DoubleConv
from torch import optim, nn
import torch
from preprocessing import C_TRAIN_PATH_IMAGES, C_TEST_PATH_IMAGES, DATA_DIR
from preprocessing import MyDataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np

MODEL_SAVE_PATH = DATA_DIR / "model_weights"

class UNet(nn.Module):
    """
    - idée: https://miro.medium.com/v2/resize:fit:720/format:webp/1*1wV5-9EVb_3KmNjfuoTqkw.png
    - sortie = (taille - kernel + 2*padding) / stride + 1
        ex pour MaxPool2d(2) : stride par défaut c'est 1

    - cf: https://github.com/uygarkurt/UNet-PyTorch/blob/main/unet.py
    [essayer de le refaire comme un grand ouaaaaaaai]
    """
    def __init__(self, in_channels=4):

        super().__init__()
        self.downconv1 = DownConv(in_channels, 64)
        self.downconv2 = DownConv(64, 128)
        self.downconv3 = DownConv(128, 256)

        self.bottle_neck = DoubleConv(256, 512)

        self.upconv3 = UpConv(512, 256)
        self.upconv2 = UpConv(256, 128)
        self.upconv1 = UpConv(128, 64)

        self.out = nn.Conv2d(64, out_channels=2, kernel_size=1) # nb de classes = 2 ==> couleurs a, b

    def forward(self, x):
        down1, p1 = self.downconv1(x)
        down2, p2 = self.downconv2(p1)
        down3, p3 = self.downconv3(p2)

        b = self.bottle_neck(p3)
        up3 = self.upconv3(b, down3)
        up2 = self.upconv2(up3, down2)
        up1 = self.upconv1(up2, down1)

        return self.out(up1)



    def fit(self, device="cuda" if torch.cuda.is_available() else "cpu", patience=3):
        LEARNING_RATE = 3e-4
        BATCH_SIZE = 32
        EPOCHS = 2
        MODEL_SAVE_PATH.mkdir(exist_ok=True)

        generator = torch.Generator().manual_seed(42)
        rng = np.random.default_rng(42)

        train_paths = np.asarray(sorted(C_TRAIN_PATH_IMAGES.iterdir()))
        idx_shuffle = rng.permutation(len(train_paths))
        eighty = int(len(train_paths) * 0.8)

        valid_paths = train_paths[idx_shuffle][:eighty]
        train_paths = train_paths[idx_shuffle][eighty:]

        train_dataset = MyDataset(train_paths, split="train", n_aug=1)
        valid_dataset = MyDataset(valid_paths, split="valid", n_aug=0)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=BATCH_SIZE,
                                      num_workers=4,
                                      shuffle=True)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                     batch_size=BATCH_SIZE,
                                     num_workers=4,
                                     shuffle=True)

        optimizer = optim.AdamW(self.parameters(), lr=LEARNING_RATE)
        criterion = nn.L1Loss()

        best_valid_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(EPOCHS):
            self.train()
            train_running_loss = 0

            for idx, (image, labels) in enumerate(tqdm(train_dataloader)):
                img = image.to(device)
                labels = labels.to(device)

                y_pred = self(img)
                optimizer.zero_grad()

                loss = criterion(y_pred, labels)
                train_running_loss += loss.item()

                loss.backward()
                optimizer.step()

            train_loss = train_running_loss / (idx + 1)

            self.eval()
            val_running_loss = 0
            with torch.no_grad():
                for idx, (image, labels) in enumerate(tqdm(valid_dataloader)):
                    img = image.to(device)
                    labels = labels.to(device)

                    y_pred = self(img)
                    loss = criterion(y_pred, labels)

                    val_running_loss += loss.item()

                val_loss = val_running_loss / (idx + 1)

            print("-"*30)
            print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
            print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
            print("-"*30)

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                wait = 0
                best_state = copy.deepcopy(self.state_dict())

            else:
                wait += 1
                if wait >= patience:
                    print(f"EARLY STOPPING : meilleure EPOCH = {epoch+1-patience}"
                          f"val loss = {best_valid_loss}")
                    break

        self.load_state_dict(best_state)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        torch.save(self.state_dict(), MODEL_SAVE_PATH / f"model{timestamp}.pt")
