import os

import h5py
import numpy as np
import streamlit as st
from PIL import Image

from helpers.streamlit_app.streamlit_computes import load_structural_tensor_images
from helpers.streamlit_app.streamlit_directories import (
    check_and_create_dir,
    is_nonempty_dir,
)
from helpers.streamlit_app.streamlit_image_loader import (
    load_images_from_dir,
    show_component_images,
    slice_viewer,
)
from src.computations.comput_features import (
    compute_average_gray_value,
    compute_structural_tensor,
    test_image_chunker,
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

    comp_type = st.selectbox(
        "Choose computation:",
        ["Average Gray Value", "Anisotropy", "Azimuthal angle", "Test Chunker"],
    )
    window_size = st.number_input("Window size:", min_value=1, value=15)
    window_radius = st.number_input("Window radius:", min_value=1, value=15)
    parallel = st.checkbox("Run in parallel?", value=False)
    max_workers = st.number_input(
        "Number of workers:", min_value=1, max_value=8, value=6
    )
    image = load_images_from_dir(data_path)

    if image is None:
        st.error("No images found in dataset path.")
        return

    match comp_type:
        case "Anisotropy":
            st.subheader("Compute anisotropy from Structural Tensor")
            if st.button("Run Computation"):
                # Look for the computed Eigen values in the results directory

                # load the eigen's and compute the anisotropy choose from drop down menu

                results = compute_structural_tensor(
                    image, window_size, parallel, max_workers
                )
                for comp, arr in results.items():
                    comp_dir = os.path.join(results_struct_dir, comp)
                    os.makedirs(comp_dir, exist_ok=True)
                    for i in range(arr.shape[0]):
                        img = Image.fromarray((arr[i]).astype(np.uint8))
                        img.save(os.path.join(comp_dir, f"{comp}_slice{i}.tiff"))
                st.success(f"Results saved in {results_struct_dir}")
            if is_nonempty_dir(results_struct_dir):
                slice_idx = st.slider("Select slice", 0, image.shape[0] - 1, 0)
                images = load_structural_tensor_images(results_struct_dir, slice_idx)
                show_component_images(images, slice_idx)

        case "Average Gray Value":
            st.subheader("Average Gray Value Computation")
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
            st.subheader("Azimuthal Angle Computation")
            st.info("This feature is under development.")

        case "Test Chunker":
            st.subheader("Test Image Chunker")
            if st.button("Run Test Chunker"):
                st.write("Testing image chunker...")
            stitched_images = test_image_chunker(
                image, window_size, parallel=parallel, max_workers=max_workers
            )
            if stitched_images is not None:
                check_and_create_dir(results_chunker_dir)
                for i in range(stitched_images.shape[0]):
                    img = Image.fromarray((stitched_images[i]).astype(np.uint8))
                    img.save(
                        os.path.join(results_chunker_dir, f"chunked_slice{i}.tiff")
                    )
                st.success(f"Chunked images saved in {results_chunker_dir}")
                index, image = slice_viewer(stitched_images, prefix="chunked_viewer")
                st.image(image, caption=f"Slice{index} of stitched images")
            else:
                st.error("Image chunker test failed.")
