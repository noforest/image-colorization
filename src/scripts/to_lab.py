from pathlib import Path
import os 
import subprocess
import tqdm
from skimage import io, color
import numpy as np

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

def convert_to_lab(image_path):
    img = io.imread(image_path)
    lab_img = color.rgb2lab(img).astype(np.float32)
    return lab_img


def save_lab(lab_img, filename, train=True):
    name = Path(filename).stem + ".npy"
    if train:
        np.save(f'{LAB_TRAIN_PATH}/{name}', lab_img)
    else:
        np.save(f'{LAB_VALID_PATH}/{name}', lab_img)


if __name__ == "__main__":
    
    c_images = os.listdir(C_TRAIN_PATH_IMAGES)
    for f in tqdm.tqdm(c_images):
        img_lab = convert_to_lab(f"{C_TRAIN_PATH_IMAGES}/{f}")
        save_lab(img_lab, f)
    c_images = os.listdir(C_VALID_PATH_IMAGES)
    for f in tqdm.tqdm(c_images):
        img_lab = convert_to_lab(f"{C_VALID_PATH_IMAGES}/{f}")
        save_lab(img_lab, f, train=False)
