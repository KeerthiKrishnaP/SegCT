import glob
import os

import numpy as np
import streamlit as st
from numpy.typing import NDArray

from helpers.streamlit_app.streamlit_directories import (
    check_and_create_dir,
    fetch_h5_files,
    save_eigen_to_h5,
    select_h5_file,
)
from helpers.streamlit_app.streamlit_image_loader import (
    load_stack_from_h5,
    load_structural_tensor,
    normalize_stack,
    preview_dataset,
    save_stack_to_h5,
    slice_viewer,
)
from src.computations.comput_features import (
    compute_structural_tensor,
    fast_eigen_computations,
    test_image_chunker,
)

# --------------------------------------------------------------------
# Cached functions
# --------------------------------------------------------------------


@st.cache_data
def load_and_normalize_images(data_path: str) -> np.ndarray | dict[str, NDArray]:
    """Load and normalize image stack from .h5 file (cached)."""
    images = load_stack_from_h5(data_path)

    return normalize_stack(images)


# --------------------------------------------------------------------
# Computation Handlers
# --------------------------------------------------------------------


def handle_structural_tensor(
    image: np.ndarray,
    result_dir: str,
    window_size: int,
    parallel: bool,
    max_workers: int,
) -> None:
    """Compute and visualize the structural tensor."""
    st.subheader("Structural Tensor Computation")
    st.session_state["file_name"] = st.text_input(
        "File name for saving results:", st.session_state.get("file_name")
    )
    if st.button("Run Computation", key="run_struct_tensor"):
        st.write("Computing structural tensor...")
        results = compute_structural_tensor(image, window_size, parallel, max_workers)
        check_and_create_dir(os.path.join(result_dir))
        save_path = os.path.join(
            result_dir,
            f"{st.session_state.get('file_name')}.h5",
        )
        save_stack_to_h5(results, save_path)
        st.success(f"Results saved in {save_path}")


def handle_eigen_computation(result_struct_dir: str, result_eigen_dir: str) -> None:
    # sourcery skip: extract-method
    """Compute and save eigenvalues and eigenvectors."""
    st.subheader("Eigenvalue Computation")

    file_name = st.text_input("File name for saving results:", "file_name")
    file = select_h5_file(fetch_h5_files(result_struct_dir))
    if st.button("Run Computation"):
        st.write("Computing eigenvalues and eigenvectors...")
        dict_components = load_structural_tensor(os.path.join(result_struct_dir, file))
        eigen_values, eigen_vectors = fast_eigen_computations(
            components=dict_components
        )
        st.write("Computation completed.")
        check_and_create_dir(result_eigen_dir)
        save_eigen_to_h5(
            result_eigen_dir, f"{file_name}.h5", eigen_values, eigen_vectors
        )
        st.success(f"Eigenvalues saved as {file_name}.h5 in {result_eigen_dir}")


def handle_test_chunker(
    image: np.ndarray,
    result_dir: str,
    window_size: int,
    parallel: bool,
    max_workers: int,
) -> None:
    """Run and visualize the test image chunker."""
    st.subheader("Test Image Chunker")

    if st.button("Run Test Chunker", key="run_chunker"):
        st.write("Running image chunker...")
        stitched_images = test_image_chunker(
            image, window_size, parallel=parallel, max_workers=max_workers
        )
        if stitched_images is not None:
            check_and_create_dir(result_dir)
            total_file_path = os.path.join(result_dir, "stitched_images.h5")
            save_stack_to_h5(stack=stitched_images, filepath=total_file_path)
            st.success(f"Chunked images saved in {result_dir}")
            index, img = slice_viewer(stitched_images, prefix="chunked_viewer")
            st.image(img, caption=f"Chunked image slice {index}")
        else:
            st.error("Image chunker test failed.")


# --------------------------------------------------------------------
#  Main Streamlit Page
# --------------------------------------------------------------------


def app() -> None:
    st.header("Segmentation and Computations")

    # --- Ensure working directory exists ---
    if "working_dir" not in st.session_state:
        st.warning("Please create or load a working directory in Page 1 first.")
        return

    base_dir = st.session_state["working_dir"]
    dataset_path = st.text_input("Enter dataset path (relative to working directory):")
    if not dataset_path:
        st.info("Please enter a dataset path.")
        return
    st.write(f"**Base directory:** {base_dir}")
    dataset_path = os.path.join(base_dir, dataset_path)
    st.write(f"**Full dataset path for .h5 files:** {dataset_path}")

    file = select_h5_file(fetch_h5_files(dataset_path))
    data_path = os.path.join(dataset_path, file)
    st.write(f"**Loaded data is {file}.h5 file @ {data_path}")

    # --- Load and preview only once ---
    if "images" not in st.session_state or st.session_state.get("last_file") != file:
        images = load_and_normalize_images(data_path)
        st.session_state["images"] = images
        st.session_state["last_file"] = file
        st.session_state["preview_shown"] = False
        st.info("Images loaded and cached successfully.")
    else:
        images = st.session_state["images"]

    if not st.session_state.get("preview_shown", False):
        if isinstance(images, np.ndarray):
            preview_dataset(images)
        st.session_state["preview_shown"] = True

    # --- Create result directories ---
    result_dirs = {
        "Structural Tensor": os.path.join(dataset_path, "results_structural_tensor"),
        "Eigen's": os.path.join(dataset_path, "results_eigen_values"),
        "Test Chunker": os.path.join(dataset_path, "results_chunker"),
    }

    # --- Computation parameters ---
    window_size = st.number_input("Window size:", min_value=1, value=15)
    window_radius = st.number_input("Window radius:", min_value=1, value=15)
    parallel = st.checkbox("Run in parallel?", value=False)
    max_workers = st.number_input(
        "Number of workers:", min_value=1, max_value=8, value=6
    )

    comp_type = st.selectbox(
        "Choose computation:",
        list(result_dirs.keys()),
    )

    # --- Dispatch handler ---
    if comp_type == "Structural Tensor":
        if isinstance(images, np.ndarray):
            handle_structural_tensor(
                images[:150, :, :],
                result_dirs["Structural Tensor"],
                window_size,
                parallel,
                max_workers,
            )

    elif comp_type == "Eigen's":
        handle_eigen_computation(
            result_dirs["Structural Tensor"], result_dirs["Eigen's"]
        )

    elif comp_type == "Test Chunker":
        if isinstance(images, np.ndarray):
            handle_test_chunker(
                images, result_dirs["Test Chunker"], window_size, parallel, max_workers
            )
