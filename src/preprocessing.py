from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from skimage import io, color
import numpy as np

CSTE_MASQUE = 0.01

class MyDataset(Dataset):
    def __init__(self, image_paths, labels, size=(256, 256), n_aug=0):
        self.image_paths = image_paths
        self.labels = labels
        self.n_aug = n_aug

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

        self.base_transform = transforms.Compose([
            transforms.Resize(size),
            normalize,
            ])


        self.aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            normalize,
            ])

    def __len__(self):
        return len(self.image_paths) * (1 + self.n_aug)

    def __getitem__(self, index):
        n = len(self.image_paths)
        real_index = index % n          # quelle image
        is_aug = index >= n             # originale ou copie augmentée

        img = io.imread(self.image_paths[real_index])
        img_LAB = color.rgb2lab(img).astype(np.float32)
        img_LAB = torch.from_numpy(img_LAB).permute(2, 0, 1)   # (3, H, W)


        transform = self.aug_transform if is_aug else self.base_transform
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



# all_train = [f"{train_dogs_dir}/{name}" for name in train_dog_fnames] + \
#         [f"{train_cats_dir}/{name}" for name in train_cat_fnames]
# all_valid = [f"{valid_dogs_dir}/{name}" for name in valid_dog_fnames] + \
#         [f"{valid_cats_dir}/{name}" for name in valid_cat_fnames]
#
# train_labels = [1] * len(train_dog_fnames) + [0] * len(train_cat_fnames)
# valid_labels = [1] * len(valid_dog_fnames) + [0] * len(valid_cat_fnames)
#
# train_dataset = MyDataset(all_train, train_labels)
# valid_dataset = MyDataset(all_valid, valid_labels)
# train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
# valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=32)
