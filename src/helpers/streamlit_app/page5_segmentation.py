import os
from collections import defaultdict

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
from src.computations.segmentation_sklearn import segment_3d_image_from_data

# --------------------------------------------------------------------
# 🔧 Utility Functions
# -------------------------------------------------------------------


def construct_feature_vector_training(data_path: str) -> dict[str, defaultdict]:
    """Construct a feature vector from given paths."""
    training_data = {
        "matrix": defaultdict(),
        "warp": defaultdict(),
        "weft": defaultdict(),
        "void": defaultdict(),
    }
    for key in training_data:
        path = os.path.join(data_path, key)
        st.write(f"Loading features for {key} from {path}")
        # avg gray value
        file = select_h5_file(
            fetch_h5_files(os.path.join(path, "results_average_gray")), for_data=key
        )
        if not os.path.exists(file):
            st.warning(f"File {file} does not exist.")
            st.write(f"Please ensure average gray value data is available for {key}.")
        gray_values = load_stack_from_h5(file)
        if gray_values is not None:
            training_data[key]["gray"] = gray_values.flatten()

        # azimuthal angle
        file = select_h5_file(
            fetch_h5_files(os.path.join(path, "results_azimuthal_angle")), for_data=key
        )
        if not os.path.exists(file):
            st.warning(f"File {file} does not exist.")
            st.write(f"Please ensure azimuthal angle data is available for {key}.")
        azmitha_agle = load_stack_from_h5(file)
        if azmitha_agle is not None:
            training_data[key]["azimuth"] = azmitha_agle.flatten

        # anisotropy
        file = select_h5_file(
            fetch_h5_files(os.path.join(path, "results_anisotropy")), for_data=key
        )
        if not os.path.exists(file):
            st.warning(f"File {file} does not exist.")
            st.write(f"Please ensure anisotropy data is available for {key}.")
        anisotropy = load_stack_from_h5(file)
        if anisotropy is not None:
            training_data[key]["anisotropy"] = anisotropy.flatten()

    return training_data


def load_feature_vector_data_to_segment(data_path: str) -> dict[str, NDArray]:
    image_features = defaultdict()
    # avg gray value
    file = select_h5_file(
        fetch_h5_files(os.path.join(data_path, "results_average_gray"))
    )
    if not os.path.exists(file):
        st.warning(f"File {file} does not exist.")
        st.write("Please ensure average gray value data is available.")
    gray_values = load_stack_from_h5(file)
    if gray_values is not None:
        image_features["gray"] = gray_values.flatten()
    # azimuthal angle
    file = select_h5_file(
        fetch_h5_files(os.path.join(data_path, "results_azimuthal_angle"))
    )
    if not os.path.exists(file):
        st.warning(f"File {file} does not exist.")
        st.write("Please ensure azimuthal angle data is available.")
    azmitha_agle = load_stack_from_h5(file)
    if azmitha_agle is not None:
        image_features["azimuth"] = azmitha_agle.flatten()
    # anisotropy
    file = select_h5_file(fetch_h5_files(os.path.join(data_path, "results_anisotropy")))
    if not os.path.exists(file):
        st.warning(f"File {file} does not exist.")
        st.write("Please ensure anisotropy data is available.")
    anisotropy = load_stack_from_h5(file)
    if anisotropy is not None:
        image_features["anisotropy"] = anisotropy.flatten()

    return image_features


def app() -> None:
    st.header("Segmentation using computed features")

    if "working_dir" not in st.session_state:
        st.warning("Please create or load a working directory in Page 1 first.")
        st.stop()  # ⛔ Stop execution until session state is ready

    base_dir = st.session_state.get("working_dir")

    main_data_path = st.text_input(
        "Enter path for data to be segmented (relative to working directory):", ""
    )
    train_data_path = st.text_input(
        "Enter path for training data (relative to working directory):", ""
    )

    # --- Wait until both inputs are filled ---
    if not main_data_path or not train_data_path:
        st.info("Please fill in both directories to continue.")
        st.stop()

    if base_dir is None:
        st.warning("Please create or load a working directory in Page 1 first.")
        st.stop()

    # Build absolute paths
    main_data_path = os.path.join(base_dir, main_data_path)
    train_data_path = os.path.join(base_dir, train_data_path)

    # --- Continue only after user clicks button ---
    if st.button("Load training data"):
        st.write(f"**Base directory:** {base_dir}")
        st.write(f"**Data to segment path:** {main_data_path}")
        st.write(f"**Training data path:** {train_data_path}")

        st.session_state["training_data"] = construct_feature_vector_training(
            train_data_path
        )
        st.success("Training data loaded successfully!")
        st.session_state["image_data"] = load_feature_vector_data_to_segment(
            main_data_path
        )
        st.success("Image features loaded successfully!")
    if st.button("Run Segmentation"):
        segmented_3d_image = segment_3d_image_from_data(
            training_data=st.session_state["training_data"],
            image_features=st.session_state["image_data"],
        )
        if segmented_3d_image is not None:
            st.success("Segmentation completed successfully!")
            st.write(f"Segmented image shape: {segmented_3d_image.shape}")
            # Save segmented image
            result_dir = os.path.join(main_data_path, "segmented_results")
            check_and_create_dir(result_dir)
            total_file_path = os.path.join(result_dir, "segmented_image.h5")
            save_stack_to_h5(stack=segmented_3d_image, filepath=total_file_path)
            st.success(f"Segmented image saved at {total_file_path}")
        else:
            st.error("Segmentation failed. Please check the logs above.")
