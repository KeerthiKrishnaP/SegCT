import os

import numpy as np
import streamlit as st
from PIL import Image

from helpers.streamlit_app.streamlit_computes import load_structural_tensor_images
from helpers.streamlit_app.streamlit_directories import (
    check_and_create_dir,
    is_nonempty_dir,
    save_eigen_to_h5,
)
from helpers.streamlit_app.streamlit_image_loader import (
    load_images_from_dir,
    load_structural_tensor_dict_from_images,
    show_component_images,
    slice_viewer,
)
from src.computations.comput_features import (
    compute_structural_tensor,
    fast_eigen_computations,
    test_image_chunker,
)


def app() -> None:  # sourcery skip: extract-method
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
        index, image = slice_viewer(sample_images, prefix="sample_viewer")
        st.image(image, caption=f"Slice {index} of  dataset")
    else:
        st.warning("No images found in the specified dataset path.")
        return

    # Directories for results
    results_struct_dir = os.path.join(data_path, "results_structural_tensor")
    results_eigen_values = os.path.join(data_path, "results_eigen_values")
    results_chunker_dir = os.path.join(data_path, "results_chunker")

    comp_type = st.selectbox(
        "Choose computation:",
        ["Structural Tensor", "Eigen's", "Test Chunker"],
    )
    window_size = st.number_input("Window size:", min_value=1, value=15)
    window_radius = st.number_input("Window radius:", min_value=1, value=15)
    parallel = st.checkbox("Run in parallel?", value=False)
    max_workers = st.number_input(
        "Number of workers:", min_value=1, max_value=8, value=6
    )
    image = load_images_from_dir(data_path)

    if image is None:
        st.error("No cropped images found in dataset path.")
        return

    match comp_type:
        case "Structural Tensor":
            st.subheader("Structural Tensor Computation")
            if st.button("Run Computation"):
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

        case "Eigen's":
            st.subheader("Eigen Values Computation")
            file_name = st.text_input(
                "File name for saving the computations:", "file_name"
            )
            if st.button("Auto fetch structural tensor"):
                if is_nonempty_dir(results_struct_dir):
                    # load images from the present path.
                    st.write(
                        f"the structural tensor is taken from {results_struct_dir}"
                    )
                else:
                    st.warning("No results for the structural tensor found")
                    st.write("compute the structural tensor first")
                # check where the data is being sourced from

            if st.button("Run Computation"):
                st.write("Computing average Eigen value...")
                dict_components = load_structural_tensor_dict_from_images(
                    results_struct_dir
                )
                eigen_values, eigen_vectors = fast_eigen_computations(
                    components=dict_components
                )
                st.write("Computation completed.")
                # function to rewrite the present directory.
                check_and_create_dir(results_eigen_values)
                save_eigen_to_h5(
                    results_eigen_values,
                    f"{file_name}.h5",
                    eigen_values,
                    eigen_vectors,
                )
                st.success(
                    f"Eigen values saved in {results_eigen_values} with file name {file_name}.h5"
                )

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
