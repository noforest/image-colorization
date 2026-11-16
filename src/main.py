#%%
from model import UNet, MODEL_SAVE_PATH
import torch
from scripts.show_model import show_image

TEMP_IMAGE_VALID = "/home/nforest001/sync-enseirb/s9/computer_vision/projet/free-computer-vision-projet/data/colored/valid/0850x2.png"
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(in_channels=4).to(device)
    pt_files = sorted(MODEL_SAVE_PATH.glob("*.pt"))
    if not pt_files:
        model.fit()
    else: # on prend le.pt le + récent
        model.load_state_dict(torch.load(pt_files[-1], map_location=device))

    #%%
    show_image(model, TEMP_IMAGE_VALID, device)



# %%
