from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from skimage import io, color
import numpy as np
from pathlib import Path

CSTE_MASQUE = 0.01

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
C_TRAIN_PATH_IMAGES = DATA_DIR / "colored" / "train"
C_TEST_PATH_IMAGES = DATA_DIR / "colored" / "test"
SIZE_CROP = 256
SIZE_CROP_ROT = 320

class PadToMultiple(transforms.Pad):
    def __init__(self, m=8, fill=0, padding_mode="reflect"):
        super().__init__(padding=0, fill=fill, padding_mode=padding_mode)
        self.m = m

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            PIL Image or Tensor: Padded image.
        """
        h, w = img.shape[-2], img.shape[-1]

        if h % self.m != 0: 
            pad_h = self.m - (h % self.m)   # combien de px à rajouter pour que ce soit un multiple de m
        else: 
            pad_h = 0                       # si multiple de 8 alors ne pas rajouter de padding inutile

        if w % self.m != 0: 
            pad_w = self.m - (w % self.m)
        else: 
            pad_w = 0

        # (left, top, right and bottom) according to the doc
        self.padding = (0, 0, pad_w, pad_h)
        return F.pad(img, self.padding, self.fill, self.padding_mode)



class MyDataset(Dataset):
    def __init__(self, image_paths, split="train", n_aug=0):
        self.image_paths = image_paths
        self.n_aug = n_aug
        self.split = split

        # NOTE: à revoir valeurs

        normalize = transforms.Normalize(
                mean=(50.0, 0.0, 0.0),
                std=(50.0, 110.0, 110.0),
                )

        """
        L  = lab[0:1] / 50.0 - 1.0   # [0,100] -> [-1,1]
        ab = lab[1:3] / 110.0        # ~[-1,1], 0 reste 0
        mask unchanged
        """

        if split == "train":
            self.base_transform = transforms.Compose([
                transforms.RandomCrop(SIZE_CROP),
                normalize,
                ])


            self.aug_transform = transforms.Compose([
                transforms.RandomCrop(SIZE_CROP_ROT),       # prend un crop plus grand exprès
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15, interpolation=InterpolationMode.BILINEAR), # la rotation va faire des triangles noirs
                transforms.CenterCrop(SIZE_CROP),           # prendre le crop centré sans les traingles noirs
                normalize,
                ])

        elif split == "valid":
            self.base_transform = transforms.Compose([
                transforms.CenterCrop(SIZE_CROP),
                normalize,
                ])
            
            self.aug_transform = None

        else:
            self.base_transform = transforms.Compose([
                PadToMultiple(8),
                normalize,
                ])

            self.aug_transform = None



    def __len__(self):
        return len(self.image_paths) * (1 + self.n_aug)

    def __getitem__(self, index):
        n = len(self.image_paths)
        real_index = index % n          # quelle image
        is_aug = index >= n             # originale ou copie augmentée

        img = io.imread(self.image_paths[real_index])
        img_LAB = color.rgb2lab(img).astype(np.float32)
        img_LAB = torch.from_numpy(img_LAB).permute(2, 0, 1)   # (3, H, W)


        transform = self.aug_transform if (is_aug and self.split == "train") else self.base_transform
        img_LAB = transform(img_LAB)

        img_L = img_LAB[0:1] # GRIS / INPUT
        img_A = img_LAB[1:2] # GROUND TRUTH
        img_B = img_LAB[2:3] # GROUND TRUTH

        # 1% des pixels à 1, le reste à 0
        masque = (torch.rand_like(img_L) < CSTE_MASQUE).float() #  normalement shape = (256, 256)

        a_hint = img_A * masque
        b_hint = img_B * masque

        x = torch.cat([img_L, a_hint, b_hint, masque], dim=0)

        return x, torch.cat([img_A, img_B], dim=0)





# train_dataset = MyDataset(os.lsdir(C_TRAIN_PATH_IMAGES))
# # valid_dataset = MyDataset(all_valid)
#
# train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
# valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=32)
