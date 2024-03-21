from typing import Optional

import numpy as np
from pydantic import NonNegativeInt
from scipy.ndimage import convolve


def compute_structual_tensor(image: np.ndarray, window_radius: NonNegativeInt):

    five_point_differentiation = np.array([1, -8, 0, 8, -1])
    intensity_1 = convolve(
        image, five_point_differentiation[np.newaxis, np.newaxis, :], mode="constant"
    )
    intensity_2 = convolve(
        image, five_point_differentiation[np.newaxis, :, np.newaxis], mode="constant"
    )
    intensity_3 = convolve(
        image, five_point_differentiation[:, np.newaxis, np.newaxis], mode="constant"
    )
    """Local structural tensor"""
    structure_tensor_11 = intensity_1**2
    structure_tensor_22 = intensity_2**2
    structure_tensor_33 = intensity_3**2
    structure_tensor_12 = intensity_1 * intensity_2
    structure_tensor_13 = intensity_1 * intensity_3
    structure_tensor_23 = intensity_2 * intensity_3

    """Compute the average structure tensor"""
    weights = np.ones(
        (2 * window_radius + 1, 2 * window_radius + 1, 2 * window_radius + 1)
    )
    weights /= np.sum(weights)

    return {
        "S11": convolve(structure_tensor_11, weights, mode="constant"),
        "S22": convolve(structure_tensor_22, weights, mode="constant"),
        "S33": convolve(structure_tensor_33, weights, mode="constant"),
        "S12": convolve(structure_tensor_12, weights, mode="constant"),
        "S13": convolve(structure_tensor_13, weights, mode="constant"),
        "S23": convolve(structure_tensor_23, weights, mode="constant"),
        "radius": window_radius,
    }
