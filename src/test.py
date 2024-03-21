import matplotlib.pyplot as plt
import numpy as np

from computations.compute import image_resize, structural_tensor

# from models.visualization import ImageViewer
from visulization.imshow_3D import show_3D_stack
from visulization.read_image_stack import load_images_from_folder

folder_path = "src/datasets/training_sets/cropped data set"
loaded_images = load_images_from_folder(folder_path)
resized_image = image_resize(image=loaded_images, compression_ratio=4)
ST = structural_tensor(image=resized_image, window_radius=4)
show_3D_stack(images=ST["S11"], number_of_slices=10)
show_3D_stack(images=ST["S22"], number_of_slices=10)
show_3D_stack(images=ST["S33"], number_of_slices=10)
plt.show()
