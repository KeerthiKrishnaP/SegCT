import os
from collections import defaultdict

import numpy as np


def make_image_to_chunks(
    image: np.ndarray, number_of_chunks: int | None, pad: bool, pad_length: int = 2
) -> dict[str, np.ndarray]:

    if number_of_chunks is None:
        if not (number_of_chunks := os.cpu_count()):
            raise KeyError("Python cannot find any CPU's for parallelization")

    direction = np.argmax(image.shape)
    number_of_voxels_in_chunk = int(image.shape[direction] / number_of_chunks)
    image_chunks = defaultdict()
    chunk_shape = [
        (0, image.shape[0]),
        (0, image.shape[1]),
        (0, image.shape[2]),
    ]

    for chunk_number in range(number_of_chunks):
        chunk_shape[direction] = (
            (
                image_chunks[str(chunk_number)].shape[direction],
                image_chunks[str(chunk_number)].shape[direction]
                + number_of_voxels_in_chunk,
            )
            if image_chunks
            else (0, number_of_voxels_in_chunk)
        )
        if chunk_number == number_of_chunks - 1 and image.shape[direction] % 2 != 0:
            chunk_shape[direction] = (
                chunk_shape[direction][0],
                chunk_shape[direction][1] + 1,
            )
        if pad:
            image_chunks[str(chunk_number + 1)] = np.pad(
                image[
                    chunk_shape[0][0] : chunk_shape[0][1],
                    chunk_shape[1][0] : chunk_shape[1][1],
                    chunk_shape[2][0] : chunk_shape[2][1],
                ],
                pad_length,
                mode="edge",
            )

    return image_chunks
