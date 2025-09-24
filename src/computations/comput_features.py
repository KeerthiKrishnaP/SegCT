from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

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
    # split image
    overlap = window_radius * 2  # ensure enough overlap for gradient and smoothing
    chunks, positions, split_axis = chunk_image_with_overlap(image, overlap)
    max_workers = 4
    # run structural tensor per chunk in parallel
    results = [] * len(chunks)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(average_gray_value, chunk, window_radius): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return stitch_chunks(results, positions, split_axis, image.shape)
