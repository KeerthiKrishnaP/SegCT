from typing import Any

import numpy as np
from numpy import intp


def chunk_image_with_overlap(
    image: np.ndarray, num_chunks: int = None, overlap: int = 10
):
    """
    Split an image into chunks along its largest axis with overlapping regions.
    Chunk size is computed automatically based on the smallest axis.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D, e.g. grayscale or RGB).
    num_chunks : int, optional
        Number of chunks to split along the largest axis.
        If None, the number of chunks is set to the ratio of largest/smallest axis.
    overlap : int
        Number of pixels to overlap between consecutive chunks.

    Returns
    -------
    chunks : list of np.ndarray
        List of overlapping image chunks.
    positions : list of tuple
        Each tuple is (start, end) index along the split axis.
    split_axis : int
        The axis along which the image was split.
    """
    # Determine split axis
    split_axis = np.argmax(image.shape[:2])  # 0=height, 1=width
    axis_len = image.shape[split_axis]
    other_axis_len = image.shape[1 - split_axis]

    # Compute number of chunks if not given
    if num_chunks is None:
        num_chunks = max(1, int(np.ceil(axis_len / other_axis_len)))

    # Compute chunk size (excluding overlap)
    chunk_size = int(np.ceil(axis_len / num_chunks))

    chunks = []
    positions = []

    start = 0
    while start < axis_len:
        end = min(start + chunk_size, axis_len)

        if split_axis == 0:  # along height
            chunk = image[max(0, start - overlap) : min(axis_len, end + overlap), :]
        else:  # along width
            chunk = image[:, max(0, start - overlap) : min(axis_len, end + overlap)]

        chunks.append(chunk)
        positions.append((start, end))
        start += chunk_size

    return chunks, positions, split_axis


def stitch_chunks(chunks, positions, split_axis, image_shape):
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
