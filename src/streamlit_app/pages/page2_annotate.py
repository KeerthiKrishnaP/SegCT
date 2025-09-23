import os

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from streamlit_app.utils.image_utilis import resize_if_needed
from streamlit_app.utils.io_untils import load_cropped_images


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
            min_value=0,
            max_value=self.image_stack.shape[0] - 1,
            value=0,
            key=f"{self.prefix}_slice_slider",
        )

    def draw_canvas(self, image: np.ndarray):
        img_pil = Image.fromarray(image)
        resized_img, scale = resize_if_needed(img_pil)
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 255, 0.2)",
            stroke_width=2,
            background_image=resized_img,
            height=resized_img.size[1],
            width=resized_img.size[0],
            drawing_mode="rect",
            key=f"{self.prefix}_canvas",
        )
        return canvas_result, scale, img_pil.size

    def save_dataset(self, rect_coords, start_idx, end_idx, dataset_name, dataset_num):
        # Create dataset directory
        dataset_dir = os.path.join(self.save_dir, dataset_name, str(dataset_num))
        os.makedirs(dataset_dir, exist_ok=True)

        x, y, w, h = rect_coords
        for i in range(start_idx, end_idx + 1):
            img = Image.fromarray(self.image_stack[i])
            cropped = img.crop((x, y, x + w, y + h))
            filename = f"roi_slice{i}.png"
            cropped.save(os.path.join(dataset_dir, filename))

        # Save metadata file
        info_path = os.path.join(dataset_dir, "roi_info.txt")
        with open(info_path, "w") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Number: {dataset_num}\n")
            f.write(f"Rectangle: x={x}, y={y}, w={w}, h={h}\n")
            f.write(f"Start slice: {start_idx}\n")
            f.write(f"End slice: {end_idx}\n")

        st.success(
            f"Saved slices {start_idx} → {end_idx} and roi_info.txt to {dataset_dir}"
        )

    def run(self):
        # Step 1: Pick a slice to draw ROI
        slice_idx = self.select_slice()
        img = self.image_stack[slice_idx]
        canvas_result, scale, _ = self.draw_canvas(img)

        # Step 2: User inputs
        st.write("### ROI Dataset Parameters")
        start_idx = st.number_input(
            "Start slice index",
            min_value=0,
            max_value=self.image_stack.shape[0] - 1,
            value=0,
            key=f"{self.prefix}_start_idx",
        )
        end_idx = st.number_input(
            "End slice index",
            min_value=0,
            max_value=self.image_stack.shape[0] - 1,
            value=self.image_stack.shape[0] - 1,
            key=f"{self.prefix}_end_idx",
        )
        dataset_name = st.text_input(
            "Dataset name (e.g., warp):", "warp", key=f"{self.prefix}_dataset_name"
        )
        dataset_num = st.number_input(
            "Dataset number:", min_value=1, value=1, key=f"{self.prefix}_dataset_num"
        )

        # Step 3: Save
        if st.button("Save Dataset", key=f"{self.prefix}_save_button"):
            data = canvas_result.json_data
            if data and "objects" in data and len(data["objects"]) > 0:
                rect = data["objects"][0]  # first rect
                # Convert back to original resolution
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


def run_page2():
    if "working_dir" not in st.session_state:
        st.warning("Please create a working directory in Page 1 first.")
        return
    cropped_stack = load_cropped_images(st.session_state["working_dir"])
    if cropped_stack is None:
        st.warning("No cropped images found. Please perform cropping in Page 1 first.")
    else:
        annotator = RectAnnotator(cropped_stack, st.session_state["working_dir"])
        annotator.run()
        annotator = RectAnnotator(cropped_stack, st.session_state["working_dir"])
        annotator.run()
