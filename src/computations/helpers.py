import os
from multiprocessing import Pool

import numpy as np


def compute_eigen(array_chunk) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eig(array_chunk.reshape(3, 3))

    return eigenvalues, eigenvectors


def parallel_compute_eigen(
    structural_tensor,
) -> list[tuple[np.ndarray, np.ndarray]]:
    chunks = np.split(structural_tensor, structural_tensor.shape[-1], axis=-1)
    with Pool() as pool:
        results = pool.map(compute_eigen, chunks)

    return results
