import glob
import os

import numpy as np
import streamlit as st
from numpy.typing import NDArray
from PIL import Image

from helpers.streamlit_app.streamlit_directories import (
    check_and_create_dir,
    is_nonempty_dir,
    save_eigen_to_h5,
)
from helpers.streamlit_app.streamlit_image_loader import (
    load_stack_from_h5,
    load_structural_tensor_images,
    normalize_stack,
    save_stack_to_h5,
    slice_viewer,
)
from src.computations.comput_features import (
    compute_structural_tensor,
    fast_eigen_computations,
    test_image_chunker,
)

# --------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------


def fetch_h5_files(data_path: str) -> list[str]:
    """Recursively find all .h5 files in directory."""

    return glob.glob(os.path.join(data_path, "**", "*.h5"), recursive=True)


def select_h5_file(h5_files: list) -> str:
    """Show file selection box for available .h5 files."""
    file_names = [os.path.basename(f) for f in h5_files]
    selected_file = st.selectbox("Select an .h5 file:", file_names)

    return h5_files[file_names.index(selected_file)]


def preview_dataset(sample_images: NDArray) -> None:
    """Display a preview image from the dataset."""
    st.write("Preview of the data set:")
    if sample_images is not None:
        index, image = slice_viewer(sample_images, prefix="sample_viewer")
        st.image(image, caption=f"Slice {index} of dataset")
        return None
    else:
        st.warning("No images found in the specified dataset path.")
        return None


def display_saved_results(result_dir: str) -> None:
    """Display computed results if available."""
    if not is_nonempty_dir(result_dir):
        return
    st.success(f"Results available in {result_dir}")
    image_dict = load_structural_tensor_images(result_dir)
    normalized_stack = normalize_stack(image_dict)

    if isinstance(normalized_stack, dict):
        # Arrange components in two rows (2 per row)
        components = list(normalized_stack.items())
        n_cols = 2
        for row_start in range(0, len(components), n_cols):
            cols = st.columns(n_cols)
            for col, (name, stack) in zip(
                cols, components[row_start : row_start + n_cols]
            ):
                if stack is not None:
                    index, image = slice_viewer(stack, prefix=f"View {name}")
                    col.image(
                        image,
                        caption=f"Slice {index} — {name}",
                        use_container_width=True,
                    )
    else:
        raise ValueError("Expected a dictionary of image stacks for structural tensor.")


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

    if st.button("Run Computation", key="run_struct_tensor"):
        st.write("Computing structural tensor...")
        results = compute_structural_tensor(image, window_size, parallel, max_workers)
        for component_name, image in results.items():
            check_and_create_dir(os.path.join(result_dir, component_name))
            save_path = os.path.join(result_dir, component_name, f"{component_name}.h5")
            save_stack_to_h5(image, save_path)
        st.success(f"Results saved in {save_path}")

    display_saved_results(result_dir)


def handle_eigen_computation(result_struct_dir: str, result_eigen_dir: str):
    """Compute and save eigenvalues and eigenvectors."""
    st.subheader("Eigenvalue Computation")

    file_name = st.text_input("File name for saving results:", "file_name")

    if st.button("Auto fetch Structural Tensor", key="fetch_struct_tensor"):
        if is_nonempty_dir(result_struct_dir):
            st.info(f"Using structural tensor data from {result_struct_dir}")
        else:
            st.warning("No structural tensor results found. Compute them first.")

    if st.button("Run Computation", key="run_eigen"):
        st.write("Computing eigenvalues and eigenvectors...")
        dict_components = load_structural_tensor_images(result_struct_dir)
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

    # --- Preview dataset ---
    images = load_stack_from_h5(data_path)
    images = normalize_stack(images)
    if images is None:
        st.warning("No images found in the specified dataset path.")
        return
    if isinstance(images, np.ndarray):
        preview_dataset(images)
    else:
        st.error("Failed to load images from the .h5 file.")
        return

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
        handle_structural_tensor(
            images,
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
        handle_test_chunker(
            images, result_dirs["Test Chunker"], window_size, parallel, max_workers
        )
