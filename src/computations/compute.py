from typing import Optional

import numpy as np
from pydantic import NonNegativeInt
from scipy.linalg import eig
from scipy.ndimage import convolve
from skimage.transform import resize


def image_resize(image: np.ndarray, compression_ratio: float) -> np.ndarray:
    print("Resizing the image")
    return resize(
        image,
        (
            int(image.shape[0] // compression_ratio),
            int(image.shape[1] // compression_ratio),
            int(image.shape[2] // compression_ratio),
        ),
        anti_aliasing=True,
    )


def average_gray_value(image, window_radius) -> np.ndarray:
    weights = np.ones(
        (
            2 * window_radius + 1,
            2 * window_radius + 1,
            2 * window_radius + 1,
        )
    )
    noramalize_weights = 0.0
    noramalize_weights /= np.sum(weights)
    average = convolve(image, noramalize_weights, mode="constant")

    return average.astype(np.float32)


def azmithal_angles(strcture_tensor, window_radius) -> np.ndarray:
    azimutal_angle = np.array([])

    return azimutal_angle.astype(np.float32)


def structure_anisotropty(strcture_tensor, window_radius) -> np.ndarray:
    structure_anisotropty = np.array([])

    return structure_anisotropty.astype(np.float32)


def eigien_values(S: dict):

    lambda1, vec1 = eig(
        [
            [S["S11"], S["S12"], S["S13"]],
            [S["S12"], S["S22"], S["S23"]],
            [S["S13"], S["S23"], S["S33"]],
        ]
    )

    idx = np.argsort(lambda1)
    lambda1 = lambda1[idx]
    vec1 = vec1[:, idx]

    beta = 1 - lambda1[:, 0] / lambda1[:, 2]
    beta[np.isnan(beta)] = 0

    if len(lambda1.shape) == 1:
        return beta
    elif lambda1.shape[1] == 1:
        return lambda1, vec1[:, :, 0], beta
    else:
        lambda1 = lambda1.T
        vec1 = vec1.swapaxes(1, 2)
        lambda1 = lambda1.squeeze()
        vec1 = vec1.squeeze()

        return lambda1, vec1[:, :, 0], vec1[:, :, 1], vec1[:, :, 2], beta


def structural_tensor(image: np.ndarray, window_radius: NonNegativeInt):
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

    return {
        "S11": convolve(
            structure_tensor_11[
                window_radius:-window_radius,
                window_radius:-window_radius,
                window_radius:-window_radius,
            ],
            weights,
            mode="constant",
        ),
        "S22": convolve(
            structure_tensor_22[
                window_radius:-window_radius,
                window_radius:-window_radius,
                window_radius:-window_radius,
            ],
            weights,
            mode="constant",
        ),
        "S33": convolve(
            structure_tensor_33[
                window_radius:-window_radius,
                window_radius:-window_radius,
                window_radius:-window_radius,
            ],
            weights,
            mode="constant",
        ),
        "S12": convolve(
            structure_tensor_12[
                window_radius:-window_radius,
                window_radius:-window_radius,
                window_radius:-window_radius,
            ],
            weights,
            mode="constant",
        ),
        "S13": convolve(
            structure_tensor_13[
                window_radius:-window_radius,
                window_radius:-window_radius,
                window_radius:-window_radius,
            ],
            weights,
            mode="constant",
        ),
        "S23": convolve(
            structure_tensor_23[
                window_radius:-window_radius,
                window_radius:-window_radius,
                window_radius:-window_radius,
            ],
            weights,
            mode="constant",
        ),
        "radius": window_radius,
    }
