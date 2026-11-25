import torch
import warnings
from pathlib import Path
import os 
import subprocess
import tqdm
from skimage import io, color
import numpy as np
from typing import Any
from numpy.typing import NDArray

ROOT_DIR = Path(__file__).resolve().parents[2]  # src/scripts/ -> src/ -> root [EMPLACEMENT DU SCRIPT]
DATA_DIR = ROOT_DIR / "data"

C_TRAIN_PATH_IMAGES = DATA_DIR / "colored" / "train"
C_VALID_PATH_IMAGES = DATA_DIR / "colored" / "valid"
# G_TRAIN_PATH_IMAGES = DATA_DIR / "grayscale" / "train"
# G_VALID_PATH_IMAGES = DATA_DIR / "grayscale" / "valid"
LAB_TRAIN_PATH = DATA_DIR / "lab" / "train"
LAB_VALID_PATH = DATA_DIR / "lab" / "valid"

subprocess.run(["mkdir", "-p", LAB_TRAIN_PATH])
subprocess.run(["mkdir", "-p", LAB_VALID_PATH])

def convert_to_lab(image_path) -> NDArray[Any]:
    """ convertit rgb en lab """
    img = io.imread(image_path)
    lab_img = color.rgb2lab(img).astype(np.float32)
    return lab_img


def convert_to_rgb(img_lab: NDArray[Any]):
    """ convertit lab en rgb """
    # couleurs prédites hors gamut sRGB => lab2rgb les ramène dans [0, 1] et prévient à chaque image : on masque ce warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Conversion from CIE-LAB")
        lab_img = color.lab2rgb(img_lab)
    return lab_img


