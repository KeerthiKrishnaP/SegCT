import itertools

import numpy as np
from numpy.typing import NDArray
from pydantic import NonNegativeInt
from scipy.ndimage import convolve

from computations.helpers import parallel_compute_eigen


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
    kernel = np.ones((kernel_size, kernel_size, kernel_size), dtype=np.float32)

    # Normalize kernel so it computes the mean
    kernel /= kernel.size

    return convolve(image.astype(np.float32), kernel, mode="constant", cval=0.0)


def azmithal_angles(strcture_tensor, window_radius) -> np.ndarray:
    azimutal_angle = np.array([])

    return azimutal_angle.astype(np.float32)


def structure_anisotropty(strcture_tensor, window_radius) -> np.ndarray:
    structure_anisotropty = np.array([])

    return structure_anisotropty.astype(np.float32)


def eigen_values_and_vectors(
    structure_tensor: dict[str, np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(structure_tensor["S11"]) > 0:
        structure_tensor_per_pixel = np.zeros(
            (len(structure_tensor["S11"].reshape(-1)), 3, 3)
        )
    else:
        raise ValueError("The structure tensor cannot be empty")

    for i, j in itertools.product(range(3), range(3)):
        string_name = "".join(["S", str(i + 1), str(j + 1)])
        structure_tensor_per_pixel[:, i, j] = structure_tensor[string_name].reshape(-1)

    return parallel_compute_eigen(structure_tensor_per_pixel)


def structural_tensor(
    image: np.ndarray,
    window_radius: NonNegativeInt,
) -> dict[str, np.ndarray]:
    image = np.pad(
        image,
        (
            (window_radius, window_radius),
            (window_radius, window_radius),
            (window_radius, window_radius),
        ),
        "linear_ramp",
    )
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
    S11 = convolve(
        structure_tensor_11[
            window_radius:-window_radius,
            window_radius:-window_radius,
            window_radius:-window_radius,
        ],
        weights,
        mode="constant",
    )
    S22 = convolve(
        structure_tensor_22[
            window_radius:-window_radius,
            window_radius:-window_radius,
            window_radius:-window_radius,
        ],
        weights,
        mode="constant",
    )
    S33 = convolve(
        structure_tensor_33[
            window_radius:-window_radius,
            window_radius:-window_radius,
            window_radius:-window_radius,
        ],
        weights,
        mode="constant",
    )
    S12 = convolve(
        structure_tensor_12[
            window_radius:-window_radius,
            window_radius:-window_radius,
            window_radius:-window_radius,
        ],
        weights,
        mode="constant",
    )
    S13 = convolve(
        structure_tensor_13[
            window_radius:-window_radius,
            window_radius:-window_radius,
            window_radius:-window_radius,
        ],
        weights,
        mode="constant",
    )
    S23 = convolve(
        structure_tensor_23[
            window_radius:-window_radius,
            window_radius:-window_radius,
            window_radius:-window_radius,
        ],
        weights,
        mode="constant",
    )

    return {
        "S11": S11,
        "S22": S22,
        "S33": S33,
        "S12": S12,
        "S13": S13,
        "S23": S23,
    }
