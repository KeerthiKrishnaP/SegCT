from typing import Any

import numpy as np
from numpy import intp
from numpy._typing import NDArray


def chunk_image_with_overlap(
    image: np.ndarray, overlap: int = 10
) -> tuple[list[Any], list[tuple[int, int]], int]:
    """
    Split an image (2D or 3D) into overlapping chunks along its longest dimension.

    - Number of chunks is computed automatically.
    - Each chunk length >= shortest dimension.
    - The last chunk may be longer if needed.

    Parameters
    ----------
    image : np.ndarray
        Input array (2D or 3D, e.g. grayscale or volumetric).
    overlap : int
        Number of voxels/pixels to overlap between consecutive chunks.

    Returns
    -------
    chunks : list[np.ndarray]
        List of overlapping image chunks.
    positions : list[tuple[int, int]]
        Start and end indices (along split axis) of each chunk.
    split_axis : int
        Axis along which the image was split.
    """
    # identify split axis
    shape = image.shape[:2] if image.ndim > 2 else image.shape
    split_axis = int(np.argmax(shape))  # longest axis
    other_axis = int(np.argmin(shape))  # shortest axis

    axis_len = image.shape[split_axis]
    min_len = image.shape[other_axis]

    # compute number of chunks
    num_chunks = int(np.ceil(axis_len / min_len))
    chunk_size = int(np.ceil(axis_len / num_chunks))

    chunks = []
    positions = []

    start = 0
    while start < axis_len:
        end = min(start + chunk_size, axis_len)

        if split_axis == 0:  # split along height
            chunk = image[max(0, start - overlap) : min(axis_len, end + overlap), ...]
        else:  # split along width
            chunk = image[..., max(0, start - overlap) : min(axis_len, end + overlap)]

        chunks.append(chunk)
        positions.append((start, end))
        start += chunk_size

    return chunks, positions, split_axis


def stitch_chunks(chunks, positions, split_axis, image_shape) -> NDArray:
    """
    Stitch back the processed chunks into a full image.

    Assumes processing did not change shape of each chunk, except possibly at overlap edges.

    Parameters
    ----------
    chunks : list of np.ndarray
        List of processed chunks.
    positions : list of tuple
        Original (start, end) positions of each chunk.
    split_axis : int
        Axis along which the image was split.
    image_shape : tuple
        Shape of the full image.

    Returns
    -------
    stitched : np.ndarray
        The reconstructed image.
    """
    stitched = np.zeros(image_shape, dtype=chunks[0].dtype)

    for chunk, (start, end) in zip(chunks, positions):
        if split_axis == 0:  # height split
            stitched[start:end, :] = chunk[
                (chunk.shape[0] - (end - start)) // 2 : (chunk.shape[0] + (end - start))
                // 2,
                :,
            ]
        else:  # width split
            stitched[:, start:end] = chunk[
                :,
                (chunk.shape[1] - (end - start)) // 2 : (chunk.shape[1] + (end - start))
                // 2,
            ]

    return stitched
