from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from skimage import io, color
import numpy as np
from pathlib import Path
from hints import make_hints, get_clicks
from utils import ROOT_DIR

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
    def __init__(self, image_paths, split="train", n_aug=0, click_user=None):
        self.image_paths = image_paths
        self.n_aug = n_aug
        self.split = split
        self.click_user = click_user

        # NOTE: à revoir valeurs

        normalize = transforms.Normalize(
                mean=(50.0, 0.0, 0.0),
                std=(50.0, 110.0, 110.0),
                )

        """
        Normalize fait (x - mean) / std sur chaque canal :
        L  = (lab[0:1] - 50) / 50    # [0,100] -> [-1,1]   (= lab/50 - 1)
        ab = (lab[1:3] - 0) / 110    # ~[-1,1], 0 reste 0
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

        # """
        # log-uniforme entre 0.01% et 1% des pixels à 1, le reste à 0
        # - Train: un nouveau masque à chaque fois donc des données variées
        # - Valid: le meme masque pour chaque image à toutes les epochs donc loss comparables
        # - Test: même idée mais pour les métriques
        # """
        # log_min, log_max = -4, -2          # log10(1e-4), log10(1e-2)
        # g = None if self.split == "train" else torch.Generator().manual_seed(index)
        # # TEST: jouer avec
        # # p = 10 ** torch.empty(1).uniform_(log_min, log_max, generator=g).item()
        #
        #
        # p = 0.001 if self.split == "test" else 10 ** torch.empty(1).uniform_(log_min, log_max, generator=g).item()
        # masque = (torch.rand(img_L.shape, generator=g) < p).float()

        """
        Clics simulés, adapté de Zhang et al. 2017 (arXiv 1705.02999) :
        - nombre de clics ~ géométrique(p), 0 inclus. Zhang: p = 1/8, soit ~7 clics par image entière.
          Nous tirons par crop 256x256 et une image fait ~10 crops: p = 1/8 donnerait ~70 clics par image
          contre ~10 pour un vrai utilisateur. On prend p = 1/4 (~3 par crop) pour couvrir la plage de test.
        - position uniforme dans le crop (la gaussienne centrée de Zhang suppose l'image entière).
        - chaque clic = carré 7x7 rempli de la couleur ab moyenne du carré, masque = 1 dessus
          (identique à make_hints à l'inférence).
        """
        n = torch.distributions.Geometric(probs=1/4).sample()

        g = None if self.split == "train" else torch.Generator().manual_seed(index)

        ab = torch.cat([img_A, img_B], dim=0)
        if self.click_user is not None:                  # click du USER
            ab_hint, masque = make_hints(ab, self.click_user) # FIX: à rajouter dans la classe Dataset
        elif self.split == "test":
            # ab_hint, masque = make_hints_automatic(ab)
            pass
        else:                                        # train / valid
            _, H, W = img_L.shape

            idx = torch.randperm(H * W, device=img_L.device)[:n]
            ys = idx // W
            xs = idx % W
            points = torch.stack([ys, xs], dim=1)   # shape [n, 2]

            ab_hint, masque = make_hints(ab, points)

        x = torch.cat([img_L, ab_hint, masque], dim=0)

        return x, torch.cat([img_A, img_B], dim=0)





# train_dataset = MyDataset(os.lsdir(C_TRAIN_PATH_IMAGES))
# # valid_dataset = MyDataset(all_valid)
#
# train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
# valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=32)
