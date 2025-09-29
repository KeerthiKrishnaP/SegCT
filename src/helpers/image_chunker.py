import numpy as np
from numpy.typing import NDArray


def chunk_image_with_overlap(image: NDArray, window: int, split_axis: int = 2):
    """
    Split image into overlapping chunks along one axis.
    Returns chunks and their (start,end) positions in the full image.
    """
    size = image.shape[split_axis]
    step = window  # effective step without overlap
    overlap = window // 2

    positions = []
    chunks = []
    start = 0
    while start < size:
        end = min(start + step, size)
        # add overlap
        chunk_start = max(0, start - overlap)
        chunk_end = min(size, end + overlap)

        sl = [slice(None)] * image.ndim
        sl[split_axis] = slice(chunk_start, chunk_end)
        chunks.append(image[tuple(sl)])
        positions.append((start, end))

        start = end  # move forward
    return chunks, positions, split_axis


def stitch_chunks(chunks, positions, split_axis: int, image_shape: tuple) -> NDArray:
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

    return stitched
