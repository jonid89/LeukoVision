import cv2
import os
from PIL import Image
import numpy as np

def is_image_valid(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"[CORRUPTED] {path} -- {e}")
        return False

def is_image_fully_readable(path):
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception as e:
        print(f"[SCRATCHED] {path} -- {e}")
        return False


def has_low_entropy(path, threshold=3.0):
    '''It measures how much varaiation in the pixel intensity distribution
    to idenitfy if one color dominates the image or not (the image has no content)'''
    try:
        with Image.open(path) as img:
            grayscale = img.convert("L")
            hist = grayscale.histogram()
            hist = np.array(hist) / sum(hist)
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            if entropy < threshold:
                print(f"[LOW ENTROPY] {path} -- entropy: {entropy:.2f}")
                return True
        return False
    except Exception as e:
        print(f"[ENTROPY ERROR] {path} -- {e}")
        return True

def has_valid_resolution(path, min_width=100, min_height=100):
    try:
        with Image.open(path) as img:
            w, h = img.size
            if w < min_width or h < min_height:
                print(f"[TOO SMALL] {path} -- {w}x{h}")
                return False
        return True
    except Exception as e:
        print(f"[RESOLUTION ERROR] {path} -- {e}")
        return False

def validate_image(path):
    return (
        is_image_valid(path)
        and is_image_fully_readable(path)
        and not has_low_entropy(path)
        and has_valid_resolution(path)
    )

def is_image_openable_cv2(path):
    img = cv2.imread(path)
    if img is None:
        print(f"[CORRUPTED - cv2] {path}")
        return False
    return True

folders=['basophil','eosinophil','erythroblast','ig','lymphocyte','monocyte','neutrophil','platelet']
base_path = "C:\data science bootcamp\data_blood_cells\data1\PBC_dataset_normal_DIB\PBC_dataset_normal_DIB"
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    print(f"\n=== Checking folder: {folder_path} ===")
    print('no. of images:',len(os.listdir(folder_path)))
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            full_path = os.path.join(folder_path, filename)
            validate_image(full_path)