from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from numpy.typing import NDArray

from computations.compute import structural_tensor
from helpers.image_chunker import chunk_image_with_overlap, stitch_chunks


def parallel_structural_tensor(
    image: NDArray,
    window_radius: int,
    parallel: bool = True,
) -> dict[Any, NDArray]:
    if parallel:
        # split image
        overlap = window_radius * 2  # +1 for gradient computation
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
    else:
        return structural_tensor(image, window_radius)
