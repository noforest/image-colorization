import os
from preprocessing import MyDataset
from scripts.to_lab import convert_to_rgb
import torch
from skimage import io
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from hints import *
from utils import add_copy_button

def denormalize_test_image(img_tensor: torch.Tensor, shape_original) -> torch.Tensor:

    """ 
    L  = (lab[0:1] + 1.0) * 50.0  # [0,100] <- [-1,1]
    ab = lab[1:3] * 110.0         
    
    - img_tensor est, au départ, "arrondie" au multiple de 8 supérieur ==> on va crop ce
    qui a été "arrondie/ajouté" par PadToMultiple(8)
    - shape_original est tirée d'un NDArray donc de la forme (H, W, ..)
    """

    h, w = img_tensor.shape[-2], img_tensor.shape[-1]   # shape de l'image selon un multiple m
    pad_h, pad_w = h - shape_original[0], w - shape_original[1] 
    img_tensor = F.crop(img_tensor, 0, 0, h-pad_h, w-pad_w)

    img_L = (img_tensor[0:1] + 1.0) * 50.0
    img_ab = img_tensor[1:3] * 110.0

    return torch.cat([img_L, img_ab], dim=0)


def show_image(model, image_path, device):

    img_rgb = io.imread(image_path)

    # 1. Construction entrée sur cpu    (/!\ on peut pas convertir en numpy quand tenseur sur GPU)
    clicks = get_clicks(image_path)
    if clicks is None:
        print("Place les clics sur la figure, puis relance cette cellule (supprimer le JSON dans clicks/ pour refaire).")
        return
    x, _ = MyDataset([image_path], split="test", click_user=clicks)[0]   # [0] important ==> permet d'executer le __getitem__(index) où index vaut donc ici 0

    img_L = x[0:1]                      # à garder pour le concat plus tard
    x_wo_mask = x[:-1]
    x_wo_mask = denormalize_test_image(x_wo_mask, img_rgb.shape).permute(1, 2, 0).numpy() # on permute car dans numpy ça doit être de la forme (H, W, channel_axis)
    x_wo_mask_rgb = convert_to_rgb(x_wo_mask)

    # 2. Puis envoie de l'entrée sur GPU
    model.eval()
    with torch.no_grad():
        x = x.unsqueeze(0).to(device)   # (B=*1*, C=4, H, W)    ==> on rajoute une dim de taille 1
        y_hat = model(x)                # (B=1, C=2, H, W)
        y_hat = y_hat.squeeze(0)        # (2, H, W)  ==> on enlève cette dim

    # 3. Concat 2 tenseurs sur CPU + GPU ==> tout mettre sur CPU
    img_pred = torch.cat([img_L, y_hat.cpu()], dim=0)
    img_pred = denormalize_test_image(img_pred, img_rgb.shape).permute(1, 2, 0).numpy()
    img_pred_rgb = convert_to_rgb(img_pred)

    # 4. Tracer les 3 figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = [
        f"Image d'origine {img_rgb.shape[:-1]}",
        f"Entrée avec qlq points de couleurs {x_wo_mask_rgb.shape[:-1]}",
        f"Image prédite colorisée {img_pred_rgb.shape[:-1]}",
    ]
    images = [img_rgb, x_wo_mask_rgb, img_pred_rgb]

    for i in range(len(axes)):
        axes[i].imshow(images[i])
        axes[i].set_title(titles[i], fontsize=9)
        axes[i].axis("off")

    plt.suptitle(os.path.basename(image_path))
    fig.tight_layout()

    plt.show()
    add_copy_button(fig)



def show_image_test(model, image_path, device):

    img_rgb = io.imread(image_path)

    # 1. Construction entrée sur cpu    (/!\ on peut pas convertir en numpy quand tenseur sur GPU)
    clicks = make_hints_automatic(image_path)

    x, _ = MyDataset([image_path], split="test", click_user=clicks)[0]   # [0] important ==> permet d'executer le __getitem__(index) où index vaut donc ici 0

    img_L = x[0:1]                      # à garder pour le concat plus tard
    x_wo_mask = x[:-1]
    x_wo_mask = denormalize_test_image(x_wo_mask, img_rgb.shape).permute(1, 2, 0).numpy() # on permute car dans numpy ça doit être de la forme (H, W, channel_axis)
    x_wo_mask_rgb = convert_to_rgb(x_wo_mask)

    # 2. Puis envoie de l'entrée sur GPU
    model.eval()
    with torch.no_grad():
        x = x.unsqueeze(0).to(device)   # (B=*1*, C=4, H, W)    ==> on rajoute une dim de taille 1
        y_hat = model(x)                # (B=1, C=2, H, W)
        y_hat = y_hat.squeeze(0)        # (2, H, W)  ==> on enlève cette dim

    # 3. Concat 2 tenseurs sur CPU + GPU ==> tout mettre sur CPU
    img_pred = torch.cat([img_L, y_hat.cpu()], dim=0)
    img_pred = denormalize_test_image(img_pred, img_rgb.shape).permute(1, 2, 0).numpy()
    img_pred_rgb = convert_to_rgb(img_pred)

    # 4. Tracer les 3 figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = [
            f"Image d'origine {img_rgb.shape[:-1]}",
            f"Entrée avec qlq points de couleurs {x_wo_mask_rgb.shape[:-1]}",
            f"Image prédite colorisée {img_pred_rgb.shape[:-1]}",
            ]
    images = [img_rgb, x_wo_mask_rgb, img_pred_rgb]

    for i in range(len(axes)):
        axes[i].imshow(images[i])
        axes[i].set_title(titles[i], fontsize=9)
        axes[i].axis("off")

        if i == 2:
            ys, xs = zip(*clicks)
            axes[i].plot(xs, ys, "r+", ms=12, mew=2)

    plt.suptitle(os.path.basename(image_path))
    fig.tight_layout()

    plt.show()
    add_copy_button(fig)
