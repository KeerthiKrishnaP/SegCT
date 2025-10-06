from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from computations.compute import (
    average_gray_value,
    eig_single_tensor,
    structural_tensor,
)
from helpers.image_chunker import chunk_image_fixed_chunks, stitch_chunks


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


def parallel_eigen_computations(
    components: dict, max_workers: int = 1
) -> tuple[ndarray[Any, Any], ndarray[Any, Any]]:
    # Flatten components → shape (N, 6)
    stacked = np.stack(
        [
            components["S11"].ravel(),
            components["S22"].ravel(),
            components["S33"].ravel(),
            components["S12"].ravel(),
            components["S13"].ravel(),
            components["S23"].ravel(),
        ],
        axis=-1,
    )
    max_workers = -1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(eig_single_tensor, stacked))

    evals, evecs = zip(*results)
    evals = np.array(evals).reshape(components["S11"].shape + (3,))
    evecs = np.array(evecs).reshape(components["S11"].shape + (3, 3))

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
