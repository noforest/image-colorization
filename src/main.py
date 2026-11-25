#%%
from preprocessing import ROOT_DIR
from pathlib import Path
import argparse
from model import UNet, MODEL_SAVE_PATH
import torch
from scripts.show_model import show_image, show_metrics
import sys
from utils import seed_everything
from IPython import get_ipython

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="réentraîne le modèle au lieu de charger le dernier .pt")
    parser.add_argument("--image", type=str, help="image que l'on veut afficher après entraînement")
    parser.add_argument("--load-model", type=str, help="poids du modèle qu'on veut utiliser")
    parser.add_argument("--test", action="store_true", help="Tester notre modèle avec les métriques PSNR, SSIM, LPIPS")

    if "ipykernel" in sys.modules:
        get_ipython().run_line_magic("matplotlib", "widget")   # fenêtre séparée pour que ginput reçoive les clics
        args, _ = parser.parse_known_args(["--load-model", "data/model_weights/1_model20261117-1613.pt", "--image", "data/colored/test/0850x2.png"])  # TEST: tester image ici
    else:
        args, _ = parser.parse_known_args()


    seed_everything()

    model = UNet(in_channels=4).to(device)
    if args.retrain:
        history = model.fit()
        show_metrics(history)
    else: # on prend le .pt le + récent
        pt_files = sorted(MODEL_SAVE_PATH.glob("*.pt"))
        if args.load_model is not None:
            model_path = Path(args.load_model)
            if not model_path.is_absolute():
                model_path = ROOT_DIR / model_path
            model.load_state_dict(torch.load(model_path, map_location=device))
        elif pt_files:
            model.load_state_dict(torch.load(pt_files[-1], map_location=device))
        else:
            sys.exit("Aucun .pt trouvé: relancer avec --retrain")

    if args.test:
        results = model.test(device)
        show_metrics(results)
    #%%
    if args.image is not None:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = ROOT_DIR / image_path

        # show_image(model, image_path, device)
        show_image(model, image_path, device, test=args.test, show_hints=True)
# %%
