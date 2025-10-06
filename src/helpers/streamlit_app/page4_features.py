import glob
import os

import numpy as np
import streamlit as st
from PIL import Image

from helpers.streamlit_app.streamlit_computes import load_structural_tensor_images
from helpers.streamlit_app.streamlit_directories import (
    check_and_create_dir,
    is_nonempty_dir,
    load_eigen_from_h5,
)
from helpers.streamlit_app.streamlit_image_loader import (
    load_images_from_dir,
    show_component_images,
    slice_viewer,
)
from src.computations.comput_features import (
    compute_anisotropy,
    compute_average_gray_value,
    compute_azimuthal_angle,
)


def app() -> None:
    st.header("Compute features from Images")

    if "working_dir" not in st.session_state:
        st.warning("Please create or load a working directory in Page 1 first.")
        return

    base_dir = st.session_state["working_dir"]
    dataset_path = st.text_input("Enter dataset path (relative to working directory):")
    data_path = os.path.join(base_dir, dataset_path)
    st.write(f"The base working directory is: {base_dir}")
    st.write(f"Full dataset path: {data_path}")
    if not os.path.exists(data_path):
        st.warning("Dataset path not found. Example: warp/1 or crops/")
        return

    # show the loaded data from data_path
    st.write("### Sample Image from Dataset")
    sample_images = load_images_from_dir(data_path)
    if sample_images is not None:
        index, image = slice_viewer(sample_images, prefix="sample_viewer")
        st.image(image, caption=f"Slice {index} of  dataset")
    else:
        st.warning("No images found in the specified dataset path.")
        return

    # Directories for results
    result_anisotropy = os.path.join(data_path, "results_anisotropy")
    results_avg_dir = os.path.join(data_path, "results_average_gray")
    results_azimuthal_angle = os.path.join(data_path, "results_azimuthal_angle")

    comp_type = st.selectbox(
        "Choose computation:",
        ["Average Gray Value", "Anisotropy", "Azimuthal angle"],
    )

    image = load_images_from_dir(data_path)

    if image is None:
        st.error("No images found in dataset path.")
        return

    match comp_type:
        case "Anisotropy":
            st.subheader("Compute anisotropy from Eigen's")
            st.write("Load precomputed Eigen values from HDF5 file.")
            if st.button("Auto fetch Eigen's"):
                path_for_eigen = os.path.join(data_path, "results_eigen_values")
                st.write(f"looking for :{path_for_eigen}")
                h5_files = glob.glob(
                    os.path.join(path_for_eigen, "**", "*.h5"), recursive=True
                )
                if not h5_files:
                    st.warning("No .h5 files found in the specified directory.")
                else:
                    file_names = [os.path.basename(f) for f in h5_files]
                    selected_file = st.selectbox("Select an .h5 file:", file_names)
                    selected_path = h5_files[file_names.index(selected_file)]
                    st.write(f"Selected file path:{selected_path}")
                    st.code(selected_path)

            if st.button("Run Computation"):
                eigen_values, eigen_vectors = load_eigen_from_h5(selected_path)
                st.write("Computing anisotropy...")
                anisotropy = compute_anisotropy(eigen_values)
                st.write("Computation completed.")
                if anisotropy is not None:
                    # function to rewrite the present directory.
                    check_and_create_dir(result_anisotropy)
                    for i in range(anisotropy.shape[0]):
                        img = Image.fromarray((anisotropy[i] * 255).astype(np.uint8))
                        img.save(
                            os.path.join(result_anisotropy, f"anisotropy_slice{i}.tiff")
                        )
            if is_nonempty_dir(result_anisotropy):
                st.success(f"Results saved in {result_anisotropy}")
                image_stack = load_images_from_dir(result_anisotropy)
                if image_stack is not None:
                    index, image = slice_viewer(image_stack, prefix="anisotropy_viewer")
                    st.image(
                        np.clip(image, 0, 255).astype(np.uint8),
                        caption=f"Anisotropy for slice {index}",
                    )

        case "Average Gray Value":
            st.subheader("Average Gray Value Computation")
            window_radius = st.number_input("Window radius:", min_value=1, value=15)
            parallel = st.checkbox("Run in parallel?", value=False)
            max_workers = st.number_input(
                "Number of workers:", min_value=1, max_value=8, value=6
            )
            if st.button("Run Computation"):
                st.write("Computing average gray value...")
                average_gray_value = compute_average_gray_value(
                    image, window_radius, parallel, max_workers
                )
                st.write("Computation completed.")
                if average_gray_value is not None:
                    # function to rewrite the present directory.
                    check_and_create_dir(results_avg_dir)
                    for i in range(average_gray_value.shape[0]):
                        img = Image.fromarray((average_gray_value[i]).astype(np.uint8))
                        img.save(
                            os.path.join(results_avg_dir, f"avg_gray_slice{i}.tiff")
                        )
            if is_nonempty_dir(results_avg_dir):
                st.success(f"Results saved in {results_avg_dir}")
                image_stack = load_images_from_dir(results_avg_dir)
                if image_stack is not None:
                    index, image = slice_viewer(image_stack, prefix="avg_gray_viewer")
                    st.image(
                        np.clip(image, 0, 255).astype(np.uint8),
                        caption=f"Average gray value for slice {index}",
                    )

        case "Azimuthal Angle":
            st.subheader("Compute Azimuthal Angle from Eigen vectors")
            st.write("Load precomputed Eigen values from HDF5 file.")
            if st.button("Auto fetch Eigen's"):
                path_for_eigen = os.path.join(data_path, "results_eigen_values")
                if h5_files := glob.glob(os.path.join(path_for_eigen, "**", "*.h5")):
                    file_names = [os.path.basename(f) for f in h5_files]
                    selected_file = st.selectbox("Select an .h5 file:", file_names)
                    selected_path = h5_files[file_names.index(selected_file)]

                    st.write("📂 Selected file path:")
                    st.code(selected_path)
                else:
                    st.warning("No .h5 files found in the specified directory.")

            if st.button("Run Computation"):
                _, eigen_vectors = load_eigen_from_h5(selected_path)
                st.write("Computing anisotropy...")
                anisotropy = compute_azimuthal_angle(eigen_vectors)
                st.write("Computation completed.")
                if anisotropy is not None:
                    # function to rewrite the present directory.
                    check_and_create_dir(results_azimuthal_angle)
                    for i in range(anisotropy.shape[0]):
                        img = Image.fromarray((anisotropy[i] * 255).astype(np.uint8))
                        img.save(
                            os.path.join(
                                results_azimuthal_angle, f"anisotropy_slice{i}.tiff"
                            )
                        )
            if is_nonempty_dir(results_azimuthal_angle):
                st.success(f"Results saved in {results_azimuthal_angle}")
                image_stack = load_images_from_dir(results_azimuthal_angle)
                if image_stack is not None:
                    index, image = slice_viewer(image_stack, prefix="anisotropy_viewer")
                    st.image(
                        np.clip(image, 0, 255).astype(np.uint8),
                        caption=f"Anisotropy for slice {index}",
                    )
