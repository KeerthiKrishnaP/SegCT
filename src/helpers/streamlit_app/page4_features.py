import os

import numpy as np
import streamlit as st
from numpy.typing import NDArray
from PIL import Image

from helpers.streamlit_app.streamlit_directories import (
    check_and_create_dir,
    fetch_h5_files,
    load_eigen_from_h5,
    select_h5_file,
)
from helpers.streamlit_app.streamlit_image_loader import (
    load_stack_from_h5,
    normalize_stack,
    save_stack_to_h5,
)
from src.computations.comput_features import (
    compute_anisotropy,
    compute_average_gray_value,
    compute_azimuthal_angle,
)

# --------------------------------------------------------------------
# 🔧 Utility Functions
# -------------------------------------------------------------------


def auto_fetch_eigen(data_path: str, session_key: str) -> None:
    """Handles loading and storing selected .h5 file path."""
    if st.button("Auto fetch Eigen's", key=f"fetch_{session_key}"):
        st.write(f"Fetching .h5 files... from {data_path}")
        h5_files = fetch_h5_files(data_path)
        if not h5_files:
            st.warning("No .h5 files found in the specified directory.")
            return
        selected_path = select_h5_file(h5_files)
        st.session_state[session_key] = selected_path
        st.write(f"📂 Selected file path: {selected_path}")


def save_image_stack(stack: np.ndarray, save_dir: str, prefix: str):
    """Save 3D stack of computed results as TIFF images."""
    check_and_create_dir(save_dir)
    for i, img_array in enumerate(stack):
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        img.save(os.path.join(save_dir, f"{prefix}_slice{i}.tiff"))


# --------------------------------------------------------------------
# Computation Handlers
# --------------------------------------------------------------------


def handle_anisotropy(result_dir: str) -> None:
    """Compute and visualize anisotropy."""
    st.subheader("Compute Anisotropy from Eigenvalues")
    if st.button("Run Computation"):
        eigen_values, _ = load_eigen_from_h5(st.session_state["eigen_path"])
        st.write("Computing anisotropy...")
        anisotropy = compute_anisotropy(eigen_values)
        st.write("Computation completed.")
        if anisotropy is not None:
            result_dir = os.path.join(
                result_dir, f"{st.session_state['file_anisotropy']}.h5"
            )
            save_stack_to_h5(anisotropy, result_dir)


def handle_average_gray_value(result_dir: str) -> None:
    """Compute and visualize average gray value."""
    st.subheader("Average Gray Value Computation")
    image_stack = load_stack_from_h5(st.session_state["image_path"])
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
            result_dir = os.path.join(result_dir, f"{st.session_state['file_avg']}.h5")
            save_stack_to_h5(avg_gray, result_dir)


def handle_azimuthal_angle(result_dir: str) -> None:
    """Compute and visualize azimuthal angle."""
    st.subheader("Compute Azimuthal Angle from Eigenvectors")
    if st.button("Run Computation"):
        _, eigen_vectors = load_eigen_from_h5(st.session_state["eigen_path"])
        st.write("Computing azimuthal angle...")
        azimuthal_angle = compute_azimuthal_angle(eigen_vectors)
        st.write("Computation completed.")
        if azimuthal_angle is not None:
            result_dir = os.path.join(
                result_dir, f"{st.session_state['file_azmi_angle']}.h5"
            )
            save_stack_to_h5(azimuthal_angle, result_dir)
    # display_results(result_dir, "azimuthal_angle")


def handle_feature_vectors(result_dir: str) -> None:
    """Compute and visualize feature vectors."""
    st.subheader("Feature Vectors Computation")
    image_stack = load_stack_from_h5(st.session_state["feature_image_path"])
    eigen_values, eigen_vectors = load_eigen_from_h5(
        st.session_state["feature_eigen_path"]
    )
    window_radius = st.number_input("Window radius:", min_value=1, value=100)
    parallel = st.checkbox("Run in parallel?", value=False)
    max_workers = st.number_input(
        "Number of workers:", min_value=1, max_value=16, value=8
    )
    feature_vector = {}
    if st.button("Run Computation"):
        st.write("Computing feature vectors...")
        avg_gray = compute_average_gray_value(
            image_stack, window_radius, parallel, max_workers
        )
        anisotropy = compute_anisotropy(eigen_values)
        azimuthal_angle = compute_azimuthal_angle(eigen_vectors)
        feature_vector = {
            "Average_Gray_Value": avg_gray,
            "Anisotropy": anisotropy,
            "Azimuthal_Angle": azimuthal_angle,
        }
        st.write("Computation completed.")
        if feature_vector:
            result_dir = os.path.join(
                result_dir, f"{st.session_state['feature_vector_name']}.h5"
            )
            save_stack_to_h5(feature_vector, result_dir)


# -------------------------------------------------------------------
# Cahce expensive operations
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_normalize_images(data: str) -> NDArray | dict[str, NDArray]:
    images = load_stack_from_h5(data)
    if images is not None:
        images = normalize_stack(images)

    return images


# --------------------------------------------------------------------
# Main Streamlit Page
# --------------------------------------------------------------------


def app() -> None:
    st.header("Compute Features from Images")

    if "working_dir" not in st.session_state:
        st.warning("Please create or load a working directory in Page 1 first.")
        return

    base_dir = st.session_state.get("working_dir")
    dataset_path = st.text_input("Enter dataset path (relative to working directory):")
    if base_dir:
        data_path = os.path.join(base_dir, dataset_path)

    st.write(f"**Base directory:** {base_dir}")
    st.write(f"**Full dataset path:** {data_path}")

    # Result directories
    result_dirs = {
        "Anisotropy": os.path.join(data_path, "results_anisotropy"),
        "Average Gray Value": os.path.join(data_path, "results_average_gray"),
        "Azimuthal Angle": os.path.join(data_path, "results_azimuthal_angle"),
        "Feature Vectors": os.path.join(data_path, "results_feature_vectors"),
    }

    comp_type = st.selectbox(
        "Choose computation:",
        list(result_dirs.keys()),
    )

    # Dispatch computation type
    if comp_type == "Anisotropy":
        path = os.path.join(data_path, "results_eigen_values")
        st.session_state["eigen_path"] = select_h5_file(fetch_h5_files(path))
        st.session_state["file_anisotropy"] = st.text_input(
            "File name for saving results:", "file_name"
        )
        handle_anisotropy(result_dirs["Anisotropy"])
    elif comp_type == "Average Gray Value":
        st.session_state["image_path"] = select_h5_file(fetch_h5_files(data_path))
        st.session_state["file_avg"] = st.text_input(
            "File name for saving results:", "file_name"
        )
        handle_average_gray_value(result_dirs["Average Gray Value"])
    elif comp_type == "Azimuthal Angle":
        path = os.path.join(data_path, "results_eigen_values")
        st.session_state["file_azmi_angle"] = st.text_input(
            "File name for saving results:", "file_name"
        )
        st.session_state["eigen_path"] = select_h5_file(fetch_h5_files(path))
        handle_azimuthal_angle(result_dirs["Azimuthal Angle"])
    elif comp_type == "Feature Vectors":
        st.write("Feature Vectors computation is currently disabled.")
    # elif comp_type == "Feature Vectors":
    #     st.session_state["feature_image_path"] = select_h5_file(
    #         fetch_h5_files(data_path)
    #     )
    #     st.session_state["feature_eigen_path"] = select_h5_file(
    #         fetch_h5_files(os.path.join(data_path, "results_eigen_values"))
    #     )
    #     st.session_state["feature_vector_name"] = st.text_input(
    #         "File name for saving results:", "file_name"
    #     )
    #     handle_feature_vectors(result_dirs["Feature Vectors"])
