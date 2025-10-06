import os
from typing import Any, Tuple

import h5py
import numpy as np
import streamlit as st


def is_nonempty_dir(path: str) -> bool:
    return os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0


def check_and_create_dir(path: str) -> None:
    if os.path.exists(path):
        import shutil

        shutil.rmtree(path)
        st.info(f"Overwriting existing directory: {path}")
    os.makedirs(path)
    st.success(f"Created directory: {path}")

    return None


def save_eigen_to_h5(
    directory: str, filename: str, evals: np.ndarray, evecs: np.ndarray
) -> None:
    """
    Save eigenvalues and eigenvectors to an HDF5 file.
    Automatically creates the directory if it doesn't exist.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)

    with h5py.File(filepath, "w") as f:
        f.create_dataset("evals", data=evals, compression="gzip", compression_opts=4)
        f.create_dataset("evecs", data=evecs, compression="gzip", compression_opts=4)

    print(f"Saved eigen data to: {filepath}")


def load_eigen_from_h5(directory: str, filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load eigenvalues and eigenvectors from an HDF5 file.
    Returns:
        (evals, evecs)
    """
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        # ✅ Use typing hint + explicit casting to keep Pylance happy
        evals = np.array(f["evals"])  # type: ignore[index]
        evecs = np.array(f["evecs"])  # type: ignore[index]

    print(f"Loaded eigen data from: {filepath}")
    return evals, evecs
