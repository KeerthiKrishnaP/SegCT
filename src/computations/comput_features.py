from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from computations.compute import (
    average_gray_value,
    structural_tensor,
)
from helpers.image_chunker import chunk_image_fixed_chunks, stitch_chunks


def compute_anisotropy(evals: np.ndarray) -> np.ndarray:
    lambda_1 = evals[..., 0]
    lambda_3 = evals[..., 2]

    beta = np.zeros_like(lambda_3)
    mask = lambda_3 > 0
    beta[mask] = 1.0 - (lambda_1[mask] / lambda_3[mask])

    return beta


def compute_azimuthal_angle(evecs: np.ndarray) -> np.ndarray:
    v = evecs[..., :, 2]  # shape (..., 3)

    vx = v[..., 0]
    vy = v[..., 1]

    # azimuthal angle (radians)
    return np.arctan2(vy, vx)


def compute_structural_tensor(
    image: NDArray, window_radius: int, parallel: bool = True, max_workers: int = 1
) -> dict[Any, NDArray]:
    if not parallel:
        return structural_tensor(image, window_radius)
    # split image
    overlap = window_radius * 2  # ensure enough overlap for gradient and smoothing
    chunks, positions, split_axis = chunk_image_fixed_chunks(
        image=image, num_chunks=max_workers, overlap=overlap
    )

    # run structural tensor per chunk in parallel
    results = [None] * len(chunks)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(structural_tensor, chunk, window_radius): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()  # type: ignore

    return {
        key: stitch_chunks(
            [res[key] for res in results],  # type: ignore
            positions,
            split_axis,
            image.shape,  # type: ignore
        )
        for key in results[0].keys()
    }


def fast_eigen_computations(
    components: dict,
) -> tuple[ndarray[Any, Any], ndarray[Any, Any]]:
    """
    components: dict with keys S11,S22,S33,S12,S13,S23, each (N,M,K)
    Returns:
        evals: (N,M,K,3)
        evecs: (N,M,K,3,3)
    """
    shape = components["S11"].shape
    n_vox = np.prod(shape)

    # Flatten components
    S11 = components["S11"].ravel()
    S22 = components["S22"].ravel()
    S33 = components["S33"].ravel()
    S12 = components["S12"].ravel()
    S13 = components["S13"].ravel()
    S23 = components["S23"].ravel()

    # Build tensor field (n_vox,3,3)
    S = np.column_stack((S11, S22, S33, S12, S13, S23))
    # Fill symmetric matrices explicitly
    S_full = np.zeros((n_vox, 3, 3), dtype=np.float64)
    S_full[:, 0, 0] = S[:, 0]
    S_full[:, 1, 1] = S[:, 1]
    S_full[:, 2, 2] = S[:, 2]
    S_full[:, 0, 1] = S_full[:, 1, 0] = S[:, 3]
    S_full[:, 0, 2] = S_full[:, 2, 0] = S[:, 4]
    S_full[:, 1, 2] = S_full[:, 2, 1] = S[:, 5]

    # ---- Eigen decomposition ----
    evals, evecs = np.linalg.eigh(S_full)

    # ---- Sort descending ----
    order = np.argsort(evals, axis=1)[:, ::-1]
    evals = np.take_along_axis(evals, order, axis=1)
    evecs = np.take_along_axis(evecs, order[:, None, :], axis=2)

    # ---- Reshape back ----
    evals = evals.reshape(shape + (3,))
    evecs = evecs.reshape(shape + (3, 3))

    return evals, evecs


def compute_average_gray_value(
    image: NDArray,
    window_radius: int,
    parallel: bool = True,
    max_workers: int = 1,
) -> NDArray:
    if not parallel:
        return average_gray_value(image, window_radius)

    # Split image into overlapping chunks
    overlap = window_radius * 2
    chunks, positions, split_axis = chunk_image_fixed_chunks(
        image=image, num_chunks=max_workers, overlap=overlap
    )
    print("images are chunked")
    # Prepare result container
    results = [None] * max_workers

    # Run average_gray_value per chunk in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(average_gray_value, chunk, window_radius): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            print("computation done, results are being written back")
            idx = futures[future]
            print(f"index is {idx}")
            results[idx] = future.result()  # type: ignore
    print("computations are over")
    # Stitch chunks back to full image
    return stitch_chunks(results, positions, split_axis, image.shape)  # type: ignore


def test_image_chunker(
    image: NDArray, window_radius: int, parallel: bool = True, max_workers: int = 1
) -> NDArray:
    if not parallel:
        return image
    chunks, positions, split_axis = chunk_image_fixed_chunks(
        image=image, num_chunks=max_workers, overlap=window_radius
    )

    print(f"Image shape: {image.shape}")
    print(f"Number of chunks: {len(chunks)}")

    stitched = stitch_chunks(chunks, positions, split_axis, image.shape)  # type: ignore
    print("Stitching successful and matches original image.")

    return stitched
