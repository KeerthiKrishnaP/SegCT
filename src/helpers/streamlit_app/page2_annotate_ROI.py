import os

import numpy as np
import streamlit as st
from PIL import Image

from helpers.streamlit_app.streamlit_image_loader import (
    draw_rectangle_canvas,
    load_images_from_dir,
)


class RectAnnotator:
    def __init__(
        self, image_stack: np.ndarray, save_dir: str, prefix="page2_annotator"
    ) -> None:
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


def app() -> None:
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
