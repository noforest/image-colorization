from scripts.to_lab import convert_to_lab, convert_to_rgb
import torch
from preprocessing import CSTE_MASQUE
from skimage import io
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def normalize_image(img_tensor, size=(256, 256)):

    """ 
    L  = lab[0:1] / 50.0 - 1.0   # [0,100] -> [-1,1]
    ab = lab[1:3] / 110.0        # ~[-1,1], 0 reste 0
    """
    # original_size = img_tensor.shape[] # NOTE: recup juste HxW

    normalize = transforms.Normalize(
            mean=(50.0, 0.0, 0.0),
            std=(50.0, 110.0, 110.0),
            )

    transformation = transforms.Compose([
        transforms.Resize(size),
        normalize,
        ])

    return transformation(img_tensor)

def denormalize_image(img_tensor, size=(256, 256)):

    """ 
    L  = (lab[0:1] + 1.0) * 50.0  # [0,100] <- [-1,1]
    ab = lab[1:3] * 110.0         
    """
    img_L = (img_tensor[0:1] + 1.0) * 50.0
    img_ab = img_tensor[1:3] * 110.0

    transformation = transforms.Compose([
        transforms.Resize(size),
        normalize,
        ])

    return transformation(img_tensor)



def show_image(model, image_path, device):

    img_rgb = io.imread(image_path)

    img_LAB = convert_to_lab(image_path)

    img_LAB = torch.from_numpy(img_LAB).permute(2, 0, 1)   # (3, H, W)

    img_L = img_LAB[0:1] # GRIS / INPUT
    img_A = img_LAB[1:2] # GROUND TRUTH
    img_B = img_LAB[2:3] # GROUND TRUTH

    masque = (torch.rand_like(img_L) < CSTE_MASQUE).float()

    a_hint = img_A * masque
    b_hint = img_B * masque

    x = torch.cat([img_L, a_hint, b_hint, masque], dim=0).to(device)

    y_hat = model(x)
    img_mask = x[:-1]
    img_mask = convert_to_rgb(img_mask)

    img_pred = torch.cat([img_L, y_hat], dim=0)
    img_pred_rgb = convert_to_rgb(img_pred)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    # for i in range(len(axes)):
    axes[0].imshow(img_mask)
    axes[1].imshow(img_rgb)
    axes[2].imshow(img_pred_rgb)

    plt.axis("off")
    plt.tight_layout()
    plt.show()



