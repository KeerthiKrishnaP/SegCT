import matplotlib.pyplot as plt
import numpy as np

from computations.compute import structural_tensor
from models.visualization import ImageViewer
from visulization.imshow_3D import show_3D_stack
from visulization.read_image_stack import load_images_from_folder

folder_path = "src/datasets/training_sets/Cropped images/"
loaded_images = load_images_from_folder(folder_path)
ImageViewer(images=loaded_images)
plt.show()
