import itertools
from multiprocessing import Pool
from typing import Optional

import numpy as np

from computations.build_structure_tensor import compute_structual_tensor
from computations.compute import (
    average_gray_value,
    azmithal_angles,
    eigien_values,
    resize,
    structure_anisotropty,
)
from models.outils import Operations


class Prallel_Process_Image:

    def __init__(
        self,
        image: np.ndarray,
        chunk_shape: tuple,
        number_of_processes: int,
        window_size: int,
        operation_name: Optional[Operations],
    ) -> None:
        self.image = image
        self.chunk_shape = chunk_shape
        self.number_of_processes = number_of_processes
        self.window_size = window_size
        self.operation_name = operation_name
        self.result = self.parallel_process_padded_image()

    def parallel_process_padded_image(self) -> np.ndarray:
        number_of_chunks = [
            self.image.shape[i] // self.chunk_shape[i] for i in range(3)
        ]
        """Process Pool"""
        with Pool(processes=self.number_of_processes) as pool:
            processed_chunks = []
            for i, j in itertools.product(
                range(number_of_chunks[0]), range(number_of_chunks[1])
            ):
                for k in range(number_of_chunks[2]):
                    slice_indices = [
                        slice(i * self.chunk_shape[0], (i + 1) * self.chunk_shape[0]),
                        slice(j * self.chunk_shape[1], (j + 1) * self.chunk_shape[1]),
                        slice(k * self.chunk_shape[2], (k + 1) * self.chunk_shape[2]),
                    ]
                    image_chunk = self.image[slice_indices]
                    padded_chunk = np.pad(image_chunk, self.window_size, mode="edge")
                    if self.operation_name == Operations.CST:
                        processed_chunks.append(
                            pool.apply_async(
                                compute_structual_tensor,
                                (padded_chunk, self.window_size),
                            )
                        )
                        processed_chunks = [result.get() for result in processed_chunks]
                        return np.concatenate(processed_chunks, axis=0)
                    elif self.operation_name == Operations.CEV:
                        processed_chunks.append(
                            pool.apply_async(
                                compute_eig_tensor, (padded_chunk, self.window_size)
                            )
                        )
