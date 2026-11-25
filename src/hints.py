import torch
import json
import sys
import numpy as np
from pathlib import Path
from utils import ROOT_DIR
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.segmentation import felzenszwalb
from scipy import ndimage as ndi
from collections import Counter
import random


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




def clicks_file(image_path):
    return CLICKS_DIR / f"{Path(image_path).stem}.json"


def pick_clicks(image_path, n=10):
    """
    Saisie des clics SANS bloquer le kernel (pour Jupyter + %matplotlib widget, ex. en SSH).
    plt.ginput bloque le kernel alors que les clics ipympl doivent justement être traités
    par ce kernel => interblocage, ginput ne rend jamais la main.
    Ici un callback enregistre le JSON à chaque clic : clic gauche = ajouter, clic droit = annuler.
    """
    f = clicks_file(image_path)
    CLICKS_DIR.mkdir(exist_ok=True)
    pts = []
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(io.imread(image_path, as_gray=True), cmap="gray")
    ax.axis("off")
    ax.set_title(f"0/{n} clics - gauche : ajouter, droit : annuler")
    marks = []

    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return
        if ev.button == 3 and pts:
            pts.pop()
            marks.pop().remove()
        elif ev.button == 1 and len(pts) < n:
            pts.append((int(ev.ydata), int(ev.xdata)))
            marks.append(ax.plot(ev.xdata, ev.ydata, "r+", ms=12, mew=2)[0])
        else:
            return
        f.write_text(json.dumps(pts))
        fin = " - terminé, relance la cellule suivante" if len(pts) == n else ""
        ax.set_title(f"{len(pts)}/{n} clics{fin}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    return pts


def get_clicks(image_path, n=10):
    """Renvoie les clics enregistrés, ou None si on vient de lancer la saisie dans Jupyter."""
    f = clicks_file(image_path)
    if f.exists():
        return [tuple(p) for p in json.loads(f.read_text())]
    if "ipykernel" in sys.modules:   # Jupyter : saisie non bloquante, on relancera après
        pick_clicks(image_path, n)
        return None
    # script lancé dans un terminal avec un affichage (TkAgg/QtAgg) : ginput marche
    plt.imshow(io.imread(image_path, as_gray=True), cmap="gray")
    plt.title(f"{n} clics max - clic droit : annuler, Entrée : terminer")
    pts = [(int(y), int(x)) for x, y in plt.ginput(n, timeout=0)]
    plt.close()
    CLICKS_DIR.mkdir(exist_ok=True)
    f.write_text(json.dumps(pts))
    return pts


def auto_clicks(image_path, n=10, epsion=0.0, gaussian=False):
    """
    Renvoie [(y, x), ...] : le centre des n plus grands segments Felzenszwalb de l'image.
    Epsilon Greedy: avec facteur espilon, on choisis une point random de distribution uniforme
    Gaussian: si true les points random sont choisi d'une distribution gausienne, si false on suit une distribution uniforme
    """
    img = io.imread(image_path)[..., :3]     # segmentation EN COULEUR (en gris, bleu et beige de même luminosité se confondent)
    H, W = img.shape[:2]
    segments = felzenszwalb(img, scale=300, sigma=0.8, min_size=int(0.002 * H * W))

    segments_sizes = Counter(segments.flatten())

    points = []
    for _ in range(n):
        if random.random() < epsion:
            if not gaussian:
                points.append((random.randint(0, H), random.randint(0, W)))
            else:
                mu = 1/2 * np.array([H, W])
                Sigma = np.array([[(H/4)**2 , 0], [0, (W/4)**2]])
                points.append(np.random.default_rng().multivariate_normal(mu, Sigma))

    for label, _ in segments_sizes.most_common():
        if len(points) == n:
            break
        # point le plus loin du bord du segment
        dist = ndi.distance_transform_edt(np.pad(segments == label, 1))[1:-1, 1:-1]   # pad : bord image = bord segment
        points.append(np.unravel_index(dist.argmax(), dist.shape))

    points = [(int(x), int(y)) for x, y in points]


    return points
