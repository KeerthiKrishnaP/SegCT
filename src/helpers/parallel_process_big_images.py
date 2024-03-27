import os
from collections import defaultdict
from multiprocessing import Pool
from typing import Optional

import numpy as np

from computations.compute import structural_tensor
from helpers.image_chunker import make_image_to_chunks
from models.outils import Operations


class PrallelProcessImage:

    def __init__(
        self,
        image: np.ndarray,
        window_size: int,
        operation_name: Optional[Operations],
    ) -> None:
        self.image = image
        self.window_size = window_size
        self.operation_name = operation_name
        self.result = self.parallel_process_image()

    def parallel_process_image(self) -> dict:
        """Get image chunks"""
        image_chunks = make_image_to_chunks(
            image=self.image,
            number_of_chunks=3,
            pad=True,
            pad_length=self.window_size,
        )
        """Create process Pool"""
        pool = Pool(processes=3)
        results = defaultdict()
        match self.operation_name:
            case Operations.CST:
                for position, image_chunk in image_chunks.items():
                    results[position] = [pool.apply_async(structural_tensor, args = (image_chunk,))]
        pool.close()
        pool.join()
        return results
