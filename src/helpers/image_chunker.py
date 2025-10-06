from typing import Any

import numpy as np
from numpy.typing import NDArray


def chunk_image_fixed_chunks(
    image: NDArray, num_chunks: int, overlap: int
) -> tuple[list[Any], list[Any], int]:
    """
    Split the image into `num_chunks` along its longest axis.
    Returns chunks and their (start,end) positions in the full image.
    """
    # Pick the longest axis
    split_axis = int(np.argmax(image.shape))
    size = image.shape[split_axis]

    # Step size with remainder handling
    base_step = size // num_chunks
    remainder = size % num_chunks  # distribute extra pixels across first chunks

    positions = []
    chunks = []
    start = 0

    for i in range(num_chunks):
        step = base_step + (1 if i < remainder else 0)
        end = start + step

        # Apply overlap
        chunk_start = max(0, start - overlap)
        chunk_end = min(size, end + overlap)

        # Build slice
        sl = [slice(None)] * image.ndim
        sl[split_axis] = slice(chunk_start, chunk_end)

        chunks.append(image[tuple(sl)])
        positions.append((start, end))

        start = end

    return chunks, positions, split_axis


def stitch_chunks(
    chunks: NDArray, positions: list, split_axis: int, image_shape: tuple
) -> NDArray:
    stitched = np.zeros(image_shape, dtype=chunks[0].dtype)

    for chunk, (start, end) in zip(chunks, positions):
        target_len = end - start
        chunk_len = chunk.shape[split_axis]
        excess = chunk_len - target_len

        # crop overlap from chunk
        trim_before = excess // 2
        trim_after = excess - trim_before
        slc_chunk = [slice(None)] * chunk.ndim
        slc_chunk[split_axis] = slice(trim_before, chunk_len - trim_after)
        chunk_cropped = chunk[tuple(slc_chunk)]

        # insert into stitched
        slc_stitched = [slice(None)] * len(image_shape)
        slc_stitched[split_axis] = slice(start, end)
        stitched[tuple(slc_stitched)] = chunk_cropped

    print("stitching done")

    return stitched
