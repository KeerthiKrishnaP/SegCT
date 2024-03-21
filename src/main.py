import matplotlib.pyplot as plt
import numpy as np

from computations.build_structure_tensor import compute_structual_tensor
from models.visualization import ImageViewer
from visulization.imshow_3D import show_3D_stack
from visulization.read_image_stack import load_images_from_folder

folder_path = "src/datasets/training_sets/cropped data set"
loaded_images = load_images_from_folder(folder_path)
structure_tensor = compute_structual_tensor(image=loaded_images[:10], window_radius=3)
viewer = ImageViewer(structure_tensor["S11"])
plt.show()
# print("structure_tensor", structure_tensor)
