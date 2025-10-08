from typing import Any

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray
from pydantic import NonNegativeInt
from scipy.ndimage import convolve1d, uniform_filter1d


def average_gray_value(image, window_radius) -> NDArray:
    """
    Compute average gray value for every voxel in a 3D image
    using a cubic neighborhood of size (2*window_radius+1)^3.

    Parameters
    ----------
    image : NDArray
        3D numpy array (grayscale image stack).
    window_radius : int
        Radius of neighborhood in voxels.

    Returns
    -------
    NDArray
        3D array of same shape as input with local average values.
    """
    # Create cubic kernel of ones
    kernel_size = 2 * window_radius + 1
    kernel = np.ones((kernel_size))

    # Normalize kernel so it computes the mean
    kernel /= kernel.size

    # Convolve image with kernel to compute local averages
    image_1 = convolve1d(image, kernel, axis=1, mode="constant")
    image_2 = convolve1d(image_1, kernel, axis=0, mode="constant")
    image_3 = convolve1d(image_2, kernel, axis=2, mode="constant")

    return image_3.astype(np.float32)


def structural_tensor(image: np.ndarray, window_radius: int) -> dict[str, np.ndarray]:
    """
    Compute the 3D structural tensor components of an image.

    Parameters
    ----------
    image : np.ndarray
        3D grayscale image (Z, Y, X).
    window_radius : int
        Radius of the local averaging window.

    Returns
    -------
    dict[str, np.ndarray]
        Structural tensor components: S11, S22, S33, S12, S13, S23
    """
    # derivative kernel (central difference)
    kernel = np.array([-1, -2, 0, 2, 1], dtype=np.float32) / 8.0

    # partial derivatives
    Ix = convolve1d(image, kernel, axis=2, mode="reflect")
    Iy = convolve1d(image, kernel, axis=1, mode="reflect")
    Iz = convolve1d(image, kernel, axis=0, mode="reflect")

    # tensor components before smoothing
    s11 = Ix * Ix
    s22 = Iy * Iy
    s33 = Iz * Iz
    s12 = Ix * Iy
    s13 = Ix * Iz
    s23 = Iy * Iz

    # local averaging (smoothing)
    kernel_size = 2 * window_radius + 1
    S11 = _get_average_over_image(s11, size=kernel_size)
    S22 = _get_average_over_image(s22, size=kernel_size)
    S33 = _get_average_over_image(s33, size=kernel_size)
    S12 = _get_average_over_image(s12, size=kernel_size)
    S13 = _get_average_over_image(s13, size=kernel_size)
    S23 = _get_average_over_image(s23, size=kernel_size)

    return {"S11": S11, "S22": S22, "S33": S33, "S12": S12, "S13": S13, "S23": S23}


def _get_average_over_image(image: NDArray, size: NonNegativeInt) -> NDArray:
    """Helper function to compute average gray value over entire image."""

    image_1 = uniform_filter1d(image, size=size, axis=0, mode="reflect")
    image_2 = uniform_filter1d(image_1, size=size, axis=1, mode="reflect")
    image_3 = uniform_filter1d(image_2, size=size, axis=2, mode="reflect")

    return image_3.astype(np.float32)
