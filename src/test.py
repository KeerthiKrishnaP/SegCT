from time import time

import matplotlib.pyplot as plt
import numpy as np

from computations.compute import image_resize, structural_tensor
from helpers.parallel_process_big_images import PrallelProcessImage
from models.outils import Operations

# from models.visualization import ImageViewer
from visulization.imshow_3D import show_3D_stack
from visulization.read_image_stack import load_images_from_folder

folder_path = "src/datasets/training_sets/Cropped images"
loaded_images = load_images_from_folder(folder_path)
resized_image = image_resize(image=loaded_images, compression_ratio=4)
start_p = time()
parallel_process_image = PrallelProcessImage(
    image=resized_image, window_size=4, operation_name=Operations.CST
)
end_p = time()
# start_n = time()
# ST = structural_tensor(image=resized_image, window_radius=4)
# end_n = time()

# print("time with prallelization", end_p - start_p)
# print("time with in normal", end_n - start_n)
# show_3D_stack(images=ST["S11"], number_of_slices=10)
# show_3D_stack(images=ST["S22"], number_of_slices=10)
# show_3D_stack(images=ST["S33"], number_of_slices=10)
# plt.show()
