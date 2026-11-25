import torch
import json
import sys
import numpy as np
from pathlib import Path
from utils import ROOT_DIR
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.segmentation import felzenszwalb
from collections import Counter


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


def make_hints_automatic(image_path, n=10):
    img = io.imread(image_path)
    img = color.rgb2gray(img)
    segments = felzenszwalb(img, scale=1000, sigma=7, min_size=100)

    segments_sizes = Counter(segments.flatten())

    points = []

    for label in segments_sizes:
        if label == 0: # label 0 pour le background
            continue
        elif len(points) == n:
            break
        points.append(np.mean(np.where(segments == label), axis=1))

    points = [(int(x), int(y)) for x, y in points]


    return points
