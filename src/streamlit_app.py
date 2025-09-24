import os
import shutil

import numpy as np
import streamlit as st
from PIL import Image

from helpers.streamlit_app.streamlit_computes import load_structural_tensor_images
from helpers.streamlit_app.streamlit_directories import is_nonempty_dir
from helpers.streamlit_app.streamlit_image_loader import (
    draw_rectangle_canvas,
    load_images_from_dir,
    show_component_images,
)
from src.computations.comput_features import parallel_structural_tensor


# ───────────────────────────────────────────
# Cropper
# ───────────────────────────────────────────
class Cropper:
    def __init__(self, image_stack: np.ndarray, save_dir: str, prefix="page1_cropper"):
        self.image_stack = image_stack
        self.save_dir = save_dir
        self.prefix = prefix
        self.crop_dir = os.path.join(save_dir, "crops", "data_stack")
        os.makedirs(self.crop_dir, exist_ok=True)

    def select_slice(self):
        return st.slider(
            "Select slice for cropping reference",
            min_value=0,
            max_value=self.image_stack.shape[0] - 1,
            value=0,
            key=f"{self.prefix}_slice_slider",
        )

    def crop_all(self, rect_coords, start_idx, end_idx):
        x, y, w, h = rect_coords
        for i in range(start_idx, end_idx + 1):
            img = Image.fromarray(self.image_stack[i])
            cropped = img.crop((x, y, x + w, y + h))
            cropped.save(os.path.join(self.crop_dir, f"cropped_slice{i}.tiff"))
        st.success(f"Saved cropped images {start_idx} → {end_idx} to {self.crop_dir}")

    def run(self):
        slice_idx = self.select_slice()
        img = self.image_stack[slice_idx]

        canvas_result, scale, _ = draw_rectangle_canvas(
            img, prefix=self.prefix, fill_color="rgba(255, 0, 0, 0.2)"
        )

        st.write("### Slice Range for Cropping")
        start_idx = st.number_input(
            "Start slice index", 0, self.image_stack.shape[0] - 1, 0
        )
        end_idx = st.number_input(
            "End slice index",
            0,
            self.image_stack.shape[0] - 1,
            self.image_stack.shape[0] - 1,
        )

        if st.button("Crop All Images", key=f"{self.prefix}_crop_button"):
            data = canvas_result.json_data
            if data and "objects" in data and len(data["objects"]) > 0:
                rect = data["objects"][0]
                x = int(rect["left"] / scale)
                y = int(rect["top"] / scale)
                w = int(rect["width"] / scale)
                h = int(rect["height"] / scale)
                self.crop_all((x, y, w, h), int(start_idx), int(end_idx))
            else:
                st.warning("Please draw a rectangle before cropping.")


# ───────────────────────────────────────────
# ROI Annotator
# ───────────────────────────────────────────
class RectAnnotator:
    def __init__(
        self, image_stack: np.ndarray, save_dir: str, prefix="page2_annotator"
    ):
        self.image_stack = image_stack
        self.save_dir = save_dir
        self.prefix = prefix

    def select_slice(self) -> int:
        return st.slider(
            "Select slice (for drawing ROI boundary)",
            0,
            self.image_stack.shape[0] - 1,
            0,
            key=f"{self.prefix}_slice_slider",
        )

    def save_dataset(self, rect_coords, start_idx, end_idx, dataset_name, dataset_num):
        dataset_dir = os.path.join(self.save_dir, dataset_name, str(dataset_num))
        os.makedirs(dataset_dir, exist_ok=True)

        x, y, w, h = rect_coords
        for i in range(start_idx, end_idx + 1):
            img = Image.fromarray(self.image_stack[i])
            cropped = img.crop((x, y, x + w, y + h))
            cropped.save(os.path.join(dataset_dir, f"roi_slice{i}.png"))

        with open(os.path.join(dataset_dir, "roi_info.txt"), "w") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Number: {dataset_num}\n")
            f.write(f"Rectangle: x={x}, y={y}, w={w}, h={h}\n")
            f.write(f"Start slice: {start_idx}\n")
            f.write(f"End slice: {end_idx}\n")

        st.success(f"Saved ROI dataset to {dataset_dir}")

    def run(self):
        slice_idx = self.select_slice()
        img = self.image_stack[slice_idx]

        canvas_result, scale, _ = draw_rectangle_canvas(
            img, prefix=self.prefix, fill_color="rgba(0, 0, 255, 0.2)"
        )

        st.write("### ROI Dataset Parameters")
        start_idx = st.number_input("Start slice", 0, self.image_stack.shape[0] - 1, 0)
        end_idx = st.number_input(
            "End slice", 0, self.image_stack.shape[0] - 1, self.image_stack.shape[0] - 1
        )
        dataset_name = st.text_input("Dataset name:", "warp")
        dataset_num = st.number_input("Dataset number:", min_value=1, value=1)

        if st.button("Save Dataset", key=f"{self.prefix}_save_button"):
            data = canvas_result.json_data
            if data and "objects" in data and len(data["objects"]) > 0:
                rect = data["objects"][0]
                x = int(rect["left"] / scale)
                y = int(rect["top"] / scale)
                w = int(rect["width"] / scale)
                h = int(rect["height"] / scale)
                self.save_dataset(
                    (x, y, w, h),
                    int(start_idx),
                    int(end_idx),
                    dataset_name,
                    dataset_num,
                )
            else:
                st.warning("Please draw a rectangle before saving.")


# ───────────────────────────────────────────
# Page 1: Crop Images
# ───────────────────────────────────────────
def page1_crop_images():
    st.header("Load & Crop Images")

    mode = st.radio(
        "Choose option:",
        ["Create New Working Directory", "Load Existing Working Directory"],
    )
    working_dir = None

    if mode == "Create New Working Directory":
        base_dir = st.text_input("Enter base path:", os.path.expanduser("~"))
        new_dir_name = st.text_input("New working directory name:", "working_output")
        working_dir = os.path.join(base_dir, new_dir_name)

        if st.button("Create Working Directory"):
            if os.path.exists(working_dir):
                shutil.rmtree(working_dir)
            os.makedirs(working_dir)
            st.session_state["working_dir"] = working_dir
            st.success(f"Created working directory: {os.path.abspath(working_dir)}")

    elif mode == "Load Existing Working Directory":
        existing_dir = st.text_input("Enter path to existing directory:")
        if st.button("Load Working Directory"):
            if os.path.exists(existing_dir) and os.path.isdir(existing_dir):
                st.session_state["working_dir"] = existing_dir
                st.success(f"Loaded working directory: {os.path.abspath(existing_dir)}")
            else:
                st.error("Invalid path.")

    if "working_dir" in st.session_state:
        uploaded_files = st.file_uploader(
            "Upload image stack",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            image_list = [np.array(Image.open(f).convert("L")) for f in uploaded_files]
            image_stack = np.stack(image_list, axis=0)
            Cropper(image_stack, st.session_state["working_dir"]).run()
    else:
        st.info("Please create or load a working directory first.")


# ───────────────────────────────────────────
# Page 3: Computations
# ───────────────────────────────────────────
def computations_page():
    st.header("Segmentation and Computations")

    if "working_dir" not in st.session_state:
        st.warning("Please create or load a working directory in Page 1 first.")
        return

    base_dir = st.session_state["working_dir"]
    dataset_path = st.text_input("Enter dataset path (relative to working directory):")
    data_path = os.path.join(base_dir, dataset_path)

    if not os.path.exists(data_path):
        st.warning("Dataset path not found. Example: warp/1 or crops/")
        return

    results_struct_dir = os.path.join(base_dir, "results_structural_tensor")
    results_struct_done = is_nonempty_dir(results_struct_dir)

    comp_type = st.selectbox(
        "Choose computation:",
        ["Structural Tensor", "Average Gray Value", "Azimuthal Angle"],
    )
    window_size = st.number_input("Window size:", min_value=1, value=15)
    parallel = st.checkbox("Run in parallel?", value=False)

    image = load_images_from_dir(data_path)
    if image is None:
        st.error("No cropped images found in dataset path.")
        return

    if comp_type == "Structural Tensor":
        st.subheader("Structural Tensor Computation")
        if st.button("Run Computation"):
            results = parallel_structural_tensor(image, window_size, parallel)
            for comp, arr in results.items():
                comp_dir = os.path.join(results_struct_dir, comp)
                os.makedirs(comp_dir, exist_ok=True)
                for i in range(arr.shape[0]):
                    img = Image.fromarray((arr[i] * 255).astype(np.uint8))
                    img.save(os.path.join(comp_dir, f"{comp}_slice{i}.png"))
            st.success(f"Results saved in {results_struct_dir}")

        if results_struct_done:
            slice_idx = st.slider("Select slice", 0, image.shape[0] - 1, 0)
            images = load_structural_tensor_images(results_struct_dir, slice_idx)
            show_component_images(images, slice_idx)


# ───────────────────────────────────────────
# Main App
# ───────────────────────────────────────────
def main():
    st.title("3D Image Stack Workflow")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Page 1: Crop Images",
            "Page 2: Annotate ROIs",
            "Page 3: Computations",
            "Page 4: Segmentation & Stats",
            "Page 5: Viewer",
        ],
    )

    if page == "Page 1: Crop Images":
        page1_crop_images()

    elif page == "Page 2: Annotate ROIs":
        st.header("Annotate Regions of Interest")
        if "working_dir" not in st.session_state:
            st.warning("Please create a working directory first.")
            return
        cropped_stack = load_images_from_dir(
            os.path.join(st.session_state["working_dir"], "crops", "data_stack")
        )
        if cropped_stack is not None:
            RectAnnotator(cropped_stack, st.session_state["working_dir"]).run()
        else:
            st.warning("No cropped images found.")

    elif page == "Page 3: Computations":
        computations_page()

    elif page == "Page 4: Segmentation & Stats":
        st.info("Segmentation page (coming soon).")

    elif page == "Page 5: Viewer":
        st.info("Viewer page (coming soon).")


if __name__ == "__main__":
    main()
