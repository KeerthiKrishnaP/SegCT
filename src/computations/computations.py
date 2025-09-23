from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from numpy import ndarray as NDArray

from computations.compute import structural_tensor
from helpers.image_chunker import chunk_image_with_overlap, stitch_chunks


def parallel_structural_tensor(
    image: np.ndarray, window_size: int, max_workers: int = 2
) -> NDArray:
    """
    Compute the structural tensor over the image in parallel.
    """
    overlap = window_size**2
    chunks, positions, split_axis = chunk_image_with_overlap(image, overlap)

    results = [] * len(chunks)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(structural_tensor, chunk, window_size): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return stitch_chunks(results, positions, split_axis, image.shape)
