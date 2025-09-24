import os

import numpy as np
import streamlit as st
from PIL import Image

from helpers.streamlit_app.streamlit_image_loader import draw_rectangle_canvas


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
