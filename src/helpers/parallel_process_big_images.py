import os
from collections import defaultdict
from multiprocessing import Pool
from typing import Optional

import numpy as np
from pydantic import NonNegativeInt

from computations.compute import structural_tensor
from helpers.image_chunker import make_image_to_chunks
from models.outils import Operations


class PrallelProcessImage:

    def __init__(
        self,
        image: np.ndarray,
        window_size: NonNegativeInt,
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
            number_of_chunks=os.cpu_count(),
            pad=True,
            pad_length=self.window_size,
        )
        """Create process Pool"""
        match self.operation_name:
            case Operations.CST:
                args_list = [(value, self.window_size) for value in image_chunks.values()]
                with Pool(processes=os.cpu_count()) as pool:
                    processed_chunks = pool.starmap(structural_tensor, args_list)

        return processed_chunks
