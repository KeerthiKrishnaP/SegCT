import glob
import os
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from helpers.streamlit_app.streamlit_directories import (
    check_and_create_dir,
    is_nonempty_dir,
    load_eigen_from_h5,
)
from helpers.streamlit_app.streamlit_image_loader import (
    load_images_from_dir,
    slice_viewer,
)
from src.computations.comput_features import (
    compute_anisotropy,
    compute_average_gray_value,
    compute_azimuthal_angle,
)

# --------------------------------------------------------------------
# 🔧 Utility Functions
# --------------------------------------------------------------------


def fetch_h5_files(data_path: str) -> list[str]:
    """Recursively find all .h5 files in results_eigen_values directory."""
    path_for_eigen = os.path.join(data_path, "results_eigen_values")

    return glob.glob(os.path.join(path_for_eigen, "**", "*.h5"), recursive=True)


def select_h5_file(h5_files: list, key: str) -> Any:
    """Show file selection box for available .h5 files."""
    file_names = [os.path.basename(f) for f in h5_files]
    selected_file = st.selectbox("Select an .h5 file:", file_names, key=key)
    return h5_files[file_names.index(selected_file)]


def auto_fetch_eigen(data_path: str, session_key: str) -> None:
    """Handles loading and storing selected .h5 file path."""
    if st.button("Auto fetch Eigen's", key=f"fetch_{session_key}"):
        h5_files = fetch_h5_files(data_path)
        if not h5_files:
            st.warning("No .h5 files found in the specified directory.")
            return
        selected_path = select_h5_file(h5_files, key=f"select_{session_key}")
        st.session_state[session_key] = selected_path
        st.write(f"📂 Selected file path: {selected_path}")


def save_image_stack(stack: np.ndarray, save_dir: str, prefix: str):
    """Save 3D stack of computed results as TIFF images."""
    check_and_create_dir(save_dir)
    for i, img_array in enumerate(stack):
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        img.save(os.path.join(save_dir, f"{prefix}_slice{i}.tiff"))


def display_results(result_dir: str, prefix: str):
    """Display results if directory is not empty."""
    if is_nonempty_dir(result_dir):
        st.success(f"✅ Results saved in {result_dir}")
        image_stack = load_images_from_dir(result_dir)
        if image_stack is not None:
            index, image = slice_viewer(image_stack, prefix=f"{prefix}_viewer")
            st.image(
                np.clip(image, 0, 255).astype(np.uint8),
                caption=f"{prefix.replace('_', ' ').capitalize()} for slice {index}",
            )


# --------------------------------------------------------------------
# Computation Handlers
# --------------------------------------------------------------------


def handle_anisotropy(data_path: str, result_dir: str) -> None:
    """Compute and visualize anisotropy."""
    st.subheader("Compute Anisotropy from Eigenvalues")
    auto_fetch_eigen(data_path, "anisotropy_path")

    if st.button("Run Computation", key="run_anisotropy"):
        if "anisotropy_path" not in st.session_state:
            st.warning("Please fetch and select an Eigen file first.")

            return

        eigen_values, _ = load_eigen_from_h5(st.session_state["anisotropy_path"])
        st.write("Computing anisotropy...")
        anisotropy = compute_anisotropy(eigen_values)
        st.write("Computation completed.")
        if anisotropy is not None:
            save_image_stack(anisotropy, result_dir, "anisotropy")

    display_results(result_dir, "anisotropy")


def handle_average_gray_value(data_path: str, result_dir: str) -> None:
    """Compute and visualize average gray value."""
    st.subheader("Average Gray Value Computation")

    image_stack = load_images_from_dir(data_path)
    if image_stack is None:
        st.error("No images found in dataset path.")
        return

    window_radius = st.number_input("Window radius:", min_value=1, value=15)
    parallel = st.checkbox("Run in parallel?", value=False)
    max_workers = st.number_input(
        "Number of workers:", min_value=1, max_value=8, value=6
    )

    if st.button("Run Computation", key="run_avg_gray"):
        st.write("Computing average gray value...")
        avg_gray = compute_average_gray_value(
            image_stack, window_radius, parallel, max_workers
        )
        st.write("Computation completed.")
        if avg_gray is not None:
            save_image_stack(avg_gray, result_dir, "avg_gray")

    display_results(result_dir, "average_gray_value")


def handle_azimuthal_angle(data_path: str, result_dir: str) -> None:
    """Compute and visualize azimuthal angle."""
    st.subheader("Compute Azimuthal Angle from Eigenvectors")

    auto_fetch_eigen(data_path, "azimuthal_path")

    if st.button("Run Computation", key="run_azimuthal"):
        if "azimuthal_path" not in st.session_state:
            st.warning("Please fetch and select an Eigen file first.")
            return

        _, eigen_vectors = load_eigen_from_h5(st.session_state["azimuthal_path"])
        st.write("Computing azimuthal angle...")
        azimuthal_angle = compute_azimuthal_angle(eigen_vectors)
        st.write("Computation completed.")
        if azimuthal_angle is not None:
            save_image_stack(azimuthal_angle, result_dir, "azimuthal_angle")

    display_results(result_dir, "azimuthal_angle")


# --------------------------------------------------------------------
# Main Streamlit Page
# --------------------------------------------------------------------


def app() -> None:
    st.header("Compute Features from Images")

    if "working_dir" not in st.session_state:
        st.warning("Please create or load a working directory in Page 1 first.")
        return

    base_dir = st.session_state["working_dir"]
    dataset_path = st.text_input("Enter dataset path (relative to working directory):")
    data_path = os.path.join(base_dir, dataset_path)

    st.write(f"**Base directory:** {base_dir}")
    st.write(f"**Full dataset path:** {data_path}")

    if not os.path.exists(data_path):
        st.warning("Dataset path not found. Example: warp/1 or crops/")
        return

    # Show sample dataset preview
    st.write("### Sample Image from Dataset")
    sample_images = load_images_from_dir(data_path)
    if sample_images is not None:
        index, image = slice_viewer(sample_images, prefix="sample_viewer")
        st.image(image, caption=f"Slice {index} of dataset")
    else:
        st.warning("No images found in the specified dataset path.")
        return

    # Result directories
    result_dirs = {
        "Anisotropy": os.path.join(data_path, "results_anisotropy"),
        "Average Gray Value": os.path.join(data_path, "results_average_gray"),
        "Azimuthal Angle": os.path.join(data_path, "results_azimuthal_angle"),
    }

    comp_type = st.selectbox(
        "Choose computation:",
        list(result_dirs.keys()),
    )

    # Dispatch computation type
    if comp_type == "Anisotropy":
        handle_anisotropy(data_path, result_dirs["Anisotropy"])
    elif comp_type == "Average Gray Value":
        handle_average_gray_value(data_path, result_dirs["Average Gray Value"])
    elif comp_type == "Azimuthal Angle":
        handle_azimuthal_angle(data_path, result_dirs["Azimuthal Angle"])
