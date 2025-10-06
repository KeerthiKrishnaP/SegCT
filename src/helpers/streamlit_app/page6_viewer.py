import os

import streamlit as st
from numpy.typing import NDArray

from helpers.streamlit_app.streamlit_image_loader import (
    load_images_from_dir,
    slice_viewer,
)
from helpers.streamlit_app.streamlit_plotters import plot_2d, plot_3d


#  Cache expensive operations
@st.cache_data(show_spinner=False)
def cached_load_images(path) -> NDArray | None:
    return load_images_from_dir(path)


def app() -> None:
    st.header("Plot the computed features")

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

    # Cached image loading
    st.write("### Sample Image from Dataset")
    sample_images = cached_load_images(data_path)
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

    # ✅ Build selection list (state-safe)
    selections = [
        st.selectbox(
            f"Choose computation {i + 1}:",
            list(result_dirs.keys()) + ["None"],
            key=f"select_{i}",
        )
        for i in range(3)
    ]

    st.session_state["selections"] = selections
    parameter_1, parameter_2, parameter_3 = selections

    chosen = [p for p in [parameter_1, parameter_2, parameter_3] if p != "None"]
    unique_chosen = list(dict.fromkeys(chosen))
    st.write(
        f"Chosen parameters: {', '.join(unique_chosen) if unique_chosen else 'None'}"
    )

    if _ := st.toggle("Show Plot"):
        if len(unique_chosen) < 2:
            st.warning("Please select at least two unique parameters.")
            st.stop()

        labels = {
            "Anisotropy": "Anisotropy (β)",
            "Average Gray Value": "Average Gray Value (G)",
            "Azimuthal Angle": "Azimuthal Angle (φ [rad])",
        }

        # Use caching for result directories too
        try:
            x = cached_load_images(result_dirs[parameter_1])
            y = cached_load_images(result_dirs[parameter_2])

            if len(unique_chosen) == 3:
                z = cached_load_images(result_dirs[parameter_3])
                fig = plot_3d(
                    x,
                    y,
                    z,
                    labels[parameter_1],
                    labels[parameter_2],
                    labels[parameter_3],
                )
            else:
                fig = plot_2d(x, y, labels[parameter_1], labels[parameter_2])

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Plotting is turned off.")
