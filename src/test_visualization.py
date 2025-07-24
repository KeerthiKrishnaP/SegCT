from time import time

from paths.image_paths import RAW_DATA

# from models.visualization import ImageViewer
from visualization.imshow_3D import show_3D_stack
from visualization.read_image_stack import load_images_from_folder

if __name__ == "__main__":
    loaded_images = load_images_from_folder(RAW_DATA, format=".tif")
    show_3D_stack(loaded_images)
