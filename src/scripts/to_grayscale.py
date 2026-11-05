import os 
from PIL import Image
import subprocess

subprocess.run(["mkdir", "-p", "../../data/grayscale/train"])
subprocess.run(["mkdir", "-p", "../../data/grayscale/valid"])

C_TRAIN_PATH_IMAGES = "../../data/colored/train"
C_VALID_PATH_IMAGES = "../../data/colored/valid"
G_TRAIN_PATH_IMAGES = "../../data/grayscale/train"
G_VALID_PATH_IMAGES = "../../data/grayscale/valid"

def convert_to_grayscale(image_path):
    img = Image.open(image_path)
    gray_img = img.convert('L')
    return gray_img

def save_image(image_gray, filename, train=True):

    img = image_gray
    if train:
        img.save(f'{G_TRAIN_PATH_IMAGES}/{filename}')
    else:
        img.save(f'{G_VALID_PATH_IMAGES}/{filename}')


if __name__ == "__main__":
    
    c_images = os.listdir(C_TRAIN_PATH_IMAGES)
    for f in c_images:
        gray_img = convert_to_grayscale(f"{C_TRAIN_PATH_IMAGES}/{f}")
        save_image(gray_img, f)
    c_images = os.listdir(C_VALID_PATH_IMAGES)
    for f in c_images:
        gray_img = convert_to_grayscale(f"{C_VALID_PATH_IMAGES}/{f}")
        save_image(gray_img, f, train=False)
