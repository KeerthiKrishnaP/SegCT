from typing import Any

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray
from pydantic import NonNegativeInt
from scipy.ndimage import convolve1d


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


def azmithal_angles(strcture_tensor, window_radius) -> np.ndarray:
    azimutal_angle = np.array([])

    return azimutal_angle.astype(np.float32)


def structure_anisotropty(strcture_tensor, window_radius) -> np.ndarray:
    structure_anisotropty = np.array([])

    return structure_anisotropty.astype(np.float32)


def eig_single_tensor(Svec) -> tuple[ndarray[Any, Any], ndarray[Any, Any]]:
    S11, S22, S33, S12, S13, S23 = Svec
    S = np.array([[S11, S12, S13], [S12, S22, S23], [S13, S23, S33]], dtype=float)
    w, V = np.linalg.eigh(S)
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]


def structural_tensor(image: np.ndarray, window_radius: int) -> dict[str, np.ndarray]:
    kernel = np.array([1, -8, 0, 8, -1], dtype=np.float32)

    # derivatives via 1D convolution
    intensity_1 = convolve1d(image, kernel, axis=2, mode="constant")
    intensity_2 = convolve1d(image, kernel, axis=1, mode="constant")
    intensity_3 = convolve1d(image, kernel, axis=0, mode="constant")

    # local tensor components
    s11 = intensity_1**2
    s22 = intensity_2**2
    s33 = intensity_3**2
    s12 = intensity_1 * intensity_2
    s13 = intensity_1 * intensity_3
    s23 = intensity_2 * intensity_3

    # smoothing via uniform filter (box kernel)
    kernel_size = 2 * window_radius + 1
    kernel = np.ones((kernel_size))

    # Normalize kernel so it computes the mean
    kernel /= kernel.size

    S11 = _get_average_over_image(s11, window_radius)
    S22 = _get_average_over_image(s22, window_radius)
    S33 = _get_average_over_image(s33, window_radius)
    S12 = _get_average_over_image(s12, window_radius)
    S13 = _get_average_over_image(s13, window_radius)
    S23 = _get_average_over_image(s23, window_radius)

    return {"S11": S11, "S22": S22, "S33": S33, "S12": S12, "S13": S13, "S23": S23}


def _get_average_over_image(image: NDArray, window_radius: NonNegativeInt) -> NDArray:
    """Helper function to compute average gray value over entire image."""
    kernel_size = 2 * window_radius + 1
    kernel = np.ones(kernel_size)

    # Normalize kernel so it computes the mean
    kernel /= kernel.size
    image_1 = convolve1d(image, kernel, axis=0, mode="constant")
    image_2 = convolve1d(image_1, kernel, axis=1, mode="constant")
    image_3 = convolve1d(image_2, kernel, axis=2, mode="constant")

    return image_3.astype(np.float32)
