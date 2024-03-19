import os

import numpy as np
from PIL import Image


def load_images_from_folder(folder_path):
    image_arrays = []
    tif_files = [
        filename for filename in os.listdir(folder_path) if filename.endswith(".tif")
    ]
    for filename in sorted(tif_files):
        image_path = os.path.join(folder_path, filename)
        try:
            image_arrays.append(np.array(Image.open(image_path)))
        except Exception:
            print("Could not load image:", image_path)

    return np.stack(image_arrays, axis=0)
