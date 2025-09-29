import os

import numpy as np
import streamlit as st
from PIL import Image

from helpers.streamlit_app.streamlit_computes import load_structural_tensor_images
from helpers.streamlit_app.streamlit_directories import is_nonempty_dir
from helpers.streamlit_app.streamlit_image_loader import (
    load_images_from_dir,
    show_component_images,
)
from src.computations.comput_features import (
    compute_average_gray_value,
    compute_structural_tensor,
    test_image_chunker,
)


def app():
    st.header("Segmentation and Computations")

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
        st.image(sample_images[0], caption="Sample Image")
    else:
        st.warning("No images found in the specified dataset path.")
        return

    results_struct_dir = os.path.join(base_dir, "results_structural_tensor")

    results_avg_dir = os.path.join(base_dir, "results_average_gray_value")

    comp_type = st.selectbox(
        "Choose computation:",
        ["Structural Tensor", "Average Gray Value", "Azimuthal Angle", "Test Chunker"],
    )
    window_size = st.number_input("Window size:", min_value=1, value=15)
    window_radius = st.number_input("Window radius:", min_value=1, value=15)
    parallel = st.checkbox("Run in parallel?", value=False)

    image = load_images_from_dir(data_path)

    if image is None:
        st.error("No cropped images found in dataset path.")
        return

    match comp_type:
        case "Structural Tensor":
            st.subheader("Structural Tensor Computation")
            if st.button("Run Computation"):
                results = compute_structural_tensor(image, window_size, parallel)
                for comp, arr in results.items():
                    comp_dir = os.path.join(results_struct_dir, comp)
                    os.makedirs(comp_dir, exist_ok=True)
                    for i in range(arr.shape[0]):
                        img = Image.fromarray((arr[i] * 255).astype(np.uint8))
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
                    image, window_radius, parallel
                )
                st.write("Computation completed.")
                if average_gray_value is not None:
                    os.makedirs(results_avg_dir, exist_ok=True)
                    for i in range(average_gray_value.shape[0]):
                        img = Image.fromarray((average_gray_value[i]).astype(np.uint32))
                        img.save(
                            os.path.join(results_avg_dir, f"avg_gray_slice{i}.tiff")
                        )
                    st.success(f"Results saved in {results_avg_dir}")
                else:
                    st.error("Average gray value computation failed.")
            if is_nonempty_dir(results_avg_dir):
                slice_idx = st.slider("Select slice", 0, image.shape[0] - 1, 0)
                images = load_images_from_dir(results_avg_dir)
                if images is not None and slice_idx < images.shape[0]:
                    st.image(
                        images[slice_idx], caption=f"Avg Gray Value Slice {slice_idx}"
                    )
                else:
                    st.error("No average gray value images found.")

        case "Azimuthal Angle":
            st.subheader("Azimuthal Angle Computation")
            st.info("This feature is under development.")

        case "Test Chunker":
            st.subheader("Test Image Chunker")
            if st.button("Run Test Chunker"):
                st.write("Testing image chunker...")
            stitched_images = test_image_chunker(image, window_size)
            if stitched_images is not None:
                st.image(
                    stitched_images,
                    caption="Stitched Image from Chunks",
                )
            else:
                st.error("Image chunker test failed.")
