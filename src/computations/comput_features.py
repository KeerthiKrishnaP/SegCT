from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
from numpy.typing import NDArray

from computations.compute import average_gray_value, structural_tensor
from helpers.image_chunker import chunk_image_with_overlap, stitch_chunks


def compute_structural_tensor(
    image: NDArray,
    window_radius: int,
    parallel: bool = True,
) -> dict[Any, NDArray]:
    if not parallel:
        return structural_tensor(image, window_radius)
    # split image
    overlap = window_radius * 2  # ensure enough overlap for gradient and smoothing
    chunks, positions, split_axis = chunk_image_with_overlap(image, overlap)
    max_workers = 4
    # run structural tensor per chunk in parallel
    results = [] * len(chunks)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(structural_tensor, chunk, window_radius): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return {
        key: stitch_chunks(
            [res[key] for res in results], positions, split_axis, image.shape
        )
        for key in results[0].keys()
    }


def compute_average_gray_value(
    image: NDArray,
    window_radius: int,
    parallel: bool = True,
) -> NDArray:
    if not parallel:
        return average_gray_value(image, window_radius)

    # Split image into overlapping chunks
    overlap = window_radius * 2
    chunks, positions, split_axis = chunk_image_with_overlap(image, overlap)
    max_workers = 4

    # Prepare result container
    results = [None] * len(chunks)

    # Run average_gray_value per chunk in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(average_gray_value, chunk, window_radius): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()  # type: ignore

    # Stitch chunks back to full image
    return stitch_chunks(results, positions, split_axis, image.shape)


def test_image_chunker(
    image: NDArray,
    window_radius: int,
    parallel: bool = True,
) -> NDArray:
    if not parallel:
        return image
    chunks, positions, split_axis = chunk_image_with_overlap(image, window_radius)
    print(f"Image shape: {image.shape}")
    print(f"Number of chunks: {len(chunks)}")

    stitched = stitch_chunks(chunks, positions, split_axis, image.shape)
    assert np.array_equal(image, stitched)
    print("Stitching successful and matches original image.")

    return stitched
