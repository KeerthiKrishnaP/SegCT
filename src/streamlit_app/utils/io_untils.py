import os

import numpy as np
from PIL import Image


def load_cropped_images(save_dir):
    crop_dir = os.path.join(save_dir, "crops", "data_stack")
    if not os.path.exists(crop_dir):
        return None
    files = sorted([f for f in os.listdir(crop_dir) if f.endswith(".tiff")])
    if not files:
        return None
    image_list = [
        np.array(Image.open(os.path.join(crop_dir, f)).convert("L")) for f in files
    ]
    return np.stack(image_list, axis=0)


def is_nonempty_dir(path):
    return os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0


def load_structural_tensor_images(results_dir, slice_idx):
    comps = ["S11", "S22", "S33", "S12", "S13", "S23"]
    images = {}
    for comp in comps:
        comp_dir = os.path.join(results_dir, comp)
        if os.path.exists(comp_dir):
            fname = f"{comp}_slice{slice_idx}.png"
            fpath = os.path.join(comp_dir, fname)
            if os.path.exists(fpath):
                images[comp] = Image.open(fpath)
    return images
