import torch
import json
from pathlib import Path
from utils import ROOT_DIR
import matplotlib.pyplot as plt
from skimage import io

CLICKS_DIR = ROOT_DIR / "clicks"   # hors de data/ pour être versionné

def make_hints(ab, points, s=3):
    """
    ab : (2, H, W) normalisé, points : [(y, x), ...]
    Pose un patch (2s+1)x(2s+1) de la couleur moyenne autour de chaque point.
    Renvoie ab_hint (2, H, W) et masque (1, H, W).
    """
    _, H, W = ab.shape
    ab_hint = torch.zeros_like(ab)
    masque = torch.zeros(1, H, W)
    for y, x in points:
        y0, y1 = max(y - s, 0), min(y + s + 1, H)   # coupe le patch au bord de l'image
        x0, x1 = max(x - s, 0), min(x + s + 1, W)
        ab_hint[:, y0:y1, x0:x1] = ab[:, y0:y1, x0:x1].mean(dim=(1, 2), keepdim=True)
        masque[:, y0:y1, x0:x1] = 1
    return ab_hint, masque




def get_clicks(image_path, n=10):
    f = CLICKS_DIR / f"{Path(image_path).stem}.json"
    if f.exists():
        return [tuple(p) for p in json.loads(f.read_text())]
    plt.imshow(io.imread(image_path, as_gray=True), cmap="gray")
    plt.title(f"{n} clics max - clic droit : annuler, Entrée : terminer")
    pts = [(int(y), int(x)) for x, y in plt.ginput(n, timeout=0)]
    plt.close()
    CLICKS_DIR.mkdir(exist_ok=True)
    f.write_text(json.dumps(pts))
    return pts