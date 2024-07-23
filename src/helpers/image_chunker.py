import os
from collections import defaultdict
from typing import Any

import numpy as np


class ImageChunker:
    def __init__(
            self,
            image: np.ndarray,
            number_of_chunks: int | None,
            pad_length: int | None,
            ) -> None:
        self.image = image
        self.number_of_chunks = self.check_number_of_chunks(number_of_chunks)
        self.pad_length = self.check_pad_length(pad_length)
        self.chunks:dict[str, np.ndarray] = self.make_image_to_chunks()

    def check_pad_length(self,pad_length: int| None) -> int:
        if pad_length is not None:
            return pad_length
        print("Warning: Additional information not provided. Using default value for Pad = 6.")
        return 6

    def check_number_of_chunks(self, number_of_chunks: int| None) -> int:
        if number_of_chunks is not None:
            return number_of_chunks
        print("Warning: Additional information not provided. Using default value for Number of chunks = 4.")
        return 4

    def make_image_to_chunks(self) -> dict[str, np.ndarray]:
        direction = np.argmax(self.image.shape)
        number_of_voxels_in_chunk = int(self.image.shape[direction] / self.number_of_chunks)
        image_chunks = defaultdict()
        chunk_shape = [
            (0, self.image.shape[0]),
            (0, self.image.shape[1]),
            (0, self.image.shape[2]),
        ]

        for chunk_number in range(self.number_of_chunks):
            chunk_shape[direction] = (
                (
                    image_chunks[str(chunk_number)].shape[direction],
                    image_chunks[str(chunk_number)].shape[direction]
                    + number_of_voxels_in_chunk,
                )
                if image_chunks
                else (0, number_of_voxels_in_chunk)
            )
            if chunk_number == self.number_of_chunks - 1 and self.image.shape[direction] % 2 != 0:
                chunk_shape[direction] = (
                    chunk_shape[direction][0],
                    chunk_shape[direction][1] + 1,
                )
            image_chunks[str(chunk_number + 1)] = np.pad(
                self.image[
                    chunk_shape[0][0] : chunk_shape[0][1],
                    chunk_shape[1][0] : chunk_shape[1][1],
                    chunk_shape[2][0] : chunk_shape[2][1],
                ],
                self.pad_length,
                mode="edge",
            )

        return image_chunks

    def build_image(self) -> np.ndarray:

        return np.ndarray([1,2])
        return np.ndarray([1,2])
