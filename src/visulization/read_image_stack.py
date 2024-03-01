import os

from PIL import Image


def load_images_from_folder(folder_path):
    images = []
    tif_files = [
        filename for filename in os.listdir(folder_path) if filename.endswith(".tif")
    ]
    for filename in sorted(tif_files):
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path)
            images.append(img)
        except Exception:
            print("Could not load image:", img_path)
    return images
