from time import time

import matplotlib.pyplot as plt
import numpy as np

from computations.compute import image_resize, structural_tensor
from helpers.parallel_process_big_images import PrallelProcessImage
from models.outils import Operations

# from models.visualization import ImageViewer
from visulization.imshow_3D import show_3D_stack
from visulization.read_image_stack import load_images_from_folder

if __name__ == "__main__":
    folder_path = "src/datasets/training_sets/Cropped images"
    loaded_images = load_images_from_folder(folder_path)
    resized_image = image_resize(image=loaded_images, compression_ratio=4)
    start_time = time()
    image_structural_tensor = structural_tensor(
        image=resized_image, window_radius=4, save=True, clear_history=False
    )
    end_time = time()
    print("time for processing", end_time - start_time)
    show_3D_stack(image_structural_tensor["S11"])
