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
    start_p = time()
    parallel_processed_image = PrallelProcessImage(
        image=resized_image, window_size=4, operation_name=Operations.CST
    ).parallel_process_image()
    end_p = time()
    start_time = time()
    structural_tensor(image=resized_image, window_radius=4)
    end_time = time()
    end_time = time()
    print("time for processingusing parallel programming", end_p - start_p)
    print("time for processing", end_time - start_time)
    show_3D_stack(parallel_processed_image[0]["S11"])
