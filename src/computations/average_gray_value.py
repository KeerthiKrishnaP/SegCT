import numpy as np
from scipy.ndimage import convolve

def average_gray_value(Image, window_radius):
    weights = np.ones((2 * window_radius + 1, 2 * window_radius + 1, 2 * window_radius + 1))
    noramalize_weights /= np.sum(weights)
    average = convolve(Image, noramalize_weights, mode='constant')

    return average.astype(np.float32)
