import matplotlib.pyplot as plt
import numpy as np

from paths.image_paths import RAW_DATA
from src.models.visualization_1 import ImageViewer
from visualization.read_image_stack import load_images_from_folder

loaded_images = load_images_from_folder(RAW_DATA, format=".tif")
ImageViewer(images=loaded_images)
plt.show()
