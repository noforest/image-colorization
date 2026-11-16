from preprocessing import MyDataset
from scripts.to_lab import convert_to_rgb
import torch
from skimage import io
import matplotlib.pyplot as plt

def denormalize_test_image(img_tensor: torch.Tensor) -> torch.Tensor:

    """ 
    L  = (lab[0:1] + 1.0) * 50.0  # [0,100] <- [-1,1]
    ab = lab[1:3] * 110.0         
    """

    # NOTE: retirer le padding pour l'inférence

    img_L = (img_tensor[0:1] + 1.0) * 50.0
    img_ab = img_tensor[1:3] * 110.0

    return torch.cat([img_L, img_ab], dim=0)


def show_image(model, image_path, device):

    img_rgb = io.imread(image_path)

    # 1. Construction entrée sur cpu    (/!\ on peut pas convertir en numpy quand tenseur sur GPU)
    x, _ = MyDataset([image_path])[0]   # [0] important ==> permet d'executer le __getitem__(index) où index vaut donc ici 0
    img_L = x[0:1]                      # à garder pour le concat plus tard
    x_wo_mask = x[:-1]
    x_wo_mask = denormalize_test_image(x_wo_mask).permute(1, 2, 0).numpy() # on permute car dans numpy ça doit être de la forme (H, W, channel_axis)
    x_wo_mask_rgb = convert_to_rgb(x_wo_mask)

    # 2. Puis envoie de l'entrée sur GPU
    model.eval()
    with torch.no_grad():
        x = x.unsqueeze(0).to(device)   # (B=*1*, C=4, H, W)    ==> on rajoute une dim de taille 1
        y_hat = model(x)                # (B=1, C=2, H, W)
        y_hat = y_hat.squeeze(0)        # (2, H, W)  ==> on enlève cette dim

    # 3. Concat 2 tenseurs sur CPU + GPU ==> tout mettre sur CPU
    img_pred = torch.cat([img_L, y_hat.cpu()], dim=0)
    img_pred = denormalize_test_image(img_pred).permute(1, 2, 0).numpy()
    img_pred_rgb = convert_to_rgb(img_pred)

    # 4. Tracer les 3 figures
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    titles = [
        f"Image d'origine {img_rgb.shape[:-1]}",
        f"Entrée avec qlq points de couleurs de l'image d'origine {x_wo_mask_rgb.shape[:-1]}",
        f"Image prédite colorisée {img_pred_rgb.shape[:-1]}",
    ]
    images = [img_rgb, x_wo_mask_rgb, img_pred_rgb]

    for i in range(len(axes)):
        axes[i].imshow(images[i])
        axes[i].set_title(titles[i])
        axes[i].axis("off")
        # axes[i].tight_layout()

    plt.show()



