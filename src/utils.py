import random
import numpy as np
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SEED = 42

def seed_everything(seed: int = SEED, deterministic: bool = False) -> None:
    """Fixe toutes les sources d'aléatoire du programme."""
    random.seed(seed)                 # module random de Python
    np.random.seed(seed)              # numpy (API globale)
    torch.manual_seed(seed)           # torch CPU + toutes les GPU

    if deterministic:                 # reproductibilité exacte sur GPU, mais plus lent
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def add_copy_button(fig):
    """
    Ajoute un bouton "presse-papier" à la barre d'outils ipympl (%matplotlib widget).
    Le bouton exécute du Python côté serveur (SSH) qui n'a pas accès au presse-papier local :
    il affiche donc sous la figure une image PNG statique, à copier par clic droit > Copier l'image.
    """
    import sys
    if "ipykernel" not in sys.modules or not hasattr(fig.canvas.toolbar, "toolitems"):
        return  # pas en mode widget : rien à faire
    from io import BytesIO
    from ipywidgets import Output
    from IPython.display import Image, display

    out = Output()
    toolbar = fig.canvas.toolbar

    def copy_png():
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        out.clear_output()
        with out:
            display(Image(data=buf.getvalue()))

    toolbar.copy_png = copy_png   # appelé par nom quand on clique le bouton
    toolbar.toolitems = list(toolbar.toolitems) + [
        ("Copy", "Image statique en dessous (clic droit > Copier l'image)", "clipboard", "copy_png")
    ]
    display(out)
