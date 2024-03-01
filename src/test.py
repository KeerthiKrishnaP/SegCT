import matplotlib.pyplot as plt
import numpy as np

from models.visualization import ImageViewer
from visulization.imshow_3D import show_3D_stack
from visulization.read_image_stack import load_images_from_folder

folder_path = "test/test_images/SlicesY_8bit"
loaded_images = load_images_from_folder(folder_path)
viewer = ImageViewer(loaded_images)
plt.show()
