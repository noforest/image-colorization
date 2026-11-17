#%%
from preprocessing import ROOT_DIR
from pathlib import Path
import argparse
from model import UNet, MODEL_SAVE_PATH
import torch
from scripts.show_model import show_image
import sys

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="réentraîne le modèle au lieu de charger le dernier .pt")
    parser.add_argument("--image", type=str, help="image que l'on veut afficher après entraînement")

    if "ipykernel" in sys.modules:
        args, _ = parser.parse_known_args(["--image", "data/colored/test/0850x2.png"])  # TEST: tester image ici
    else:
        args, _ = parser.parse_known_args()



    model = UNet(in_channels=4).to(device)
    if args.retrain:
        model.fit()
    else: # on prend le .pt le + récent
        pt_files = sorted(MODEL_SAVE_PATH.glob("*.pt"))
        model.load_state_dict(torch.load(pt_files[-1], map_location=device))

    #%%
    if args.image is not None:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = ROOT_DIR / image_path

        show_image(model, image_path, device)
# %%