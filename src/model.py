import copy
from unet_pieces import UpConv
from unet_pieces import DownConv
from unet_pieces import DoubleConv
from torch import optim, nn
import torch
from preprocessing import C_TRAIN_PATH_IMAGES, DATA_DIR, C_TEST_PATH_IMAGES
from preprocessing import MyDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import numpy as np
from utils import SEED
from torch.nn import Tanh
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from preprocessing import denormalize_test_image
from scripts.to_lab import convert_to_rgb

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
        self.activation = Tanh()

    def forward(self, x):
        down1, p1 = self.downconv1(x)
        down2, p2 = self.downconv2(p1)
        down3, p3 = self.downconv3(p2)

        b = self.bottle_neck(p3)
        up3 = self.upconv3(b, down3)
        up2 = self.upconv2(up3, down2)
        up1 = self.upconv1(up2, down1)

        out = self.out(up1)
        return self.activation(out)



    def fit(self, device="cuda" if torch.cuda.is_available() else "cpu", patience=3):
        LEARNING_RATE = 3e-4
        BATCH_SIZE = 32
        EPOCHS = 30
        MODEL_SAVE_PATH.mkdir(exist_ok=True)

        train_paths = np.asarray(sorted(C_TRAIN_PATH_IMAGES.iterdir()))

        rng = np.random.default_rng(SEED) # rng dédié: split identique quoi qu'il se passe avant (np.random.seed n'affecte pas default_rng)
        idx_shuffle = rng.permutation(len(train_paths))
        eighty = int(len(train_paths) * 0.8)

        valid_paths = train_paths[idx_shuffle][eighty:]
        train_paths = train_paths[idx_shuffle][:eighty]

        train_dataset = MyDataset(train_paths, split="train", n_aug=1, p_geom=1/4)
        valid_dataset = MyDataset(valid_paths, split="valid", n_aug=0)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=BATCH_SIZE,
                                      num_workers=4,
                                      shuffle=True)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                     batch_size=BATCH_SIZE,
                                     num_workers=4,
                                     shuffle=False)

        optimizer = optim.AdamW(self.parameters(), lr=LEARNING_RATE)
        criterion = nn.L1Loss()

        best_valid_loss = float("inf")
        best_state = None
        wait = 0

        history = {"train_loss": [], "val_loss": [],
                   "val_psnr": [], "val_ssim": [], "val_lpips": []}

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

            psnr_batch = 0
            ssim_batch = 0
            lpips_batch = 0

            self.eval()
            val_running_loss = 0
            with torch.no_grad():
                for idx, (image, labels) in enumerate(tqdm(valid_dataloader)):
                    img = image.to(device)
                    labels = labels.to(device)

                    y_pred = self(img)
                    loss = criterion(y_pred, labels)
                    val_running_loss += loss.item()

                    psnr = PeakSignalNoiseRatio()
                    psnr_batch += psnr(y_pred, labels)

                    ssim = StructuralSimilarityIndexMeasure()
                    ssim_batch += ssim(y_pred, labels)

                    lpips = LearnedPerceptualImagePatchSimilarity()
                    lpips_batch += lpips(y_pred, labels)

                val_loss = val_running_loss / (idx + 1)


            print("-"*30)
            print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
            print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
            print(f"psnr EPOCH {epoch+1}: {psnr_batch / (idx + 1)}")
            print(f"ssim EPOCH {epoch+1}: {psnr_batch / (idx + 1)}")
            print(f"lpips EPOCH {epoch+1}: {psnr_batch / (idx + 1)}")
            print("-"*30)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["psnr"].append(psnr_batch / (idx + 1))
            history["ssim"].append(ssim_batch / (idx + 1))
            history["lpips"].append(lpips_batch / (idx + 1))


            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                wait = 0
                best_state = copy.deepcopy(self.state_dict())

            else:
                wait += 1
                if wait >= patience:
                    print(f"EARLY STOPPING : meilleure EPOCH = {epoch+1-patience} | "
                          f"val loss = {best_valid_loss}")
                    break

        self.load_state_dict(best_state)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        torch.save(self.state_dict(), MODEL_SAVE_PATH / f"model{timestamp}.pt")

        return history


    def test(self, device="cuda" if torch.cuda.is_available() else "cpu"):

        LEARNING_RATE = 3e-4
        BATCH_SIZE = 32
        criterion = nn.L1Loss()

        test_paths = np.asarray(sorted(C_TEST_PATH_IMAGES.iterdir()))
        test_dataset = MyDataset(test_paths, split="test", n_aug=0, n_clicks=10)
        test_dataloader = DataLoader(dataset=test_dataset,
                                      batch_size=BATCH_SIZE,
                                      num_workers=4,
                                      shuffle=False)

        self.to(device)
        self.eval()

        test_running_loss = 0
        psnr_batch = 0
        ssim_batch = 0
        lpips_batch = 0
        testr = []

        history = {"test_loss": [], "psnr": [], "ssim": [], "lpips": []}
        with torch.no_grad():

            for idx, (image, labels) in enumerate(tqdm(test_dataloader)):
                img = image.to(device)
                labels = labels.to(device)

                y_pred = self(img)
                loss = criterion(y_pred, labels)
                test_running_loss += loss.item()

                psnr = PeakSignalNoiseRatio()
                psnr_batch += psnr(y_pred, labels)

                ssim = StructuralSimilarityIndexMeasure()
                ssim_batch += ssim(y_pred, labels)

                lpips = LearnedPerceptualImagePatchSimilarity()
                lpips_batch += lpips(y_pred, labels)

            history["test_loss"] = test_running_loss / (idx + 1)
            history["psnr"] = psnr_batch / (idx + 1)
            history["ssim"] = ssim_batch / (idx + 1)
            history["lpips"] = lpips_batch / (idx + 1)

        print("-"*30)
        print(f"loss moyenne = {np.mean(history['test_loss'])}")
        print(f"psnr moyen = {np.mean(history['psnr'])}")
        print(f"ssim moyen = {np.mean(history['ssim'])}")
        print(f"lpips moyen = {np.mean(history['lpips'])}")
        print("-"*30)
        return history

