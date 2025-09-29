import os
import shutil

import numpy as np
import streamlit as st
from PIL import Image

from helpers.streamlit_app.streamlit_image_loader import (
    draw_rectangle_canvas,
    load_images_from_dir,
    slice_viewer,
)


class Cropper:
    def __init__(self, image_stack, save_dir, prefix="page1_cropper"):
        self.image_stack = image_stack
        self.save_dir = save_dir
        self.prefix = prefix
        self.crop_dir = os.path.join(save_dir, "crops", "data_stack")
        os.makedirs(self.crop_dir, exist_ok=True)

    def crop_all(self, rect_coords, start_idx, end_idx):
        x, y, w, h = rect_coords
        for i in range(start_idx, end_idx + 1):
            img = Image.fromarray(self.image_stack[i])
            cropped = img.crop((x, y, x + w, y + h))
            cropped.save(os.path.join(self.crop_dir, f"cropped_slice{i}.tiff"))

    def run(self):
        slice_idx, img = slice_viewer(self.image_stack, prefix=self.prefix)

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
                self._draw_rectangle(data, scale, start_idx, end_idx)
            else:
                st.warning("Please draw a rectangle before cropping.")

    # TODO Rename this here and in `run`
    def _draw_rectangle(self, data, scale, start_idx, end_idx):
        rect = data["objects"][0]
        x = int(rect["left"] / scale)
        y = int(rect["top"] / scale)
        w = int(rect["width"] / scale)
        h = int(rect["height"] / scale)
        self.crop_all((x, y, w, h), int(start_idx), int(end_idx))
        st.success(f"Saved cropped images {start_idx} → {end_idx} to {self.crop_dir}")


def app():
    st.header("Load & Crop Images")

    mode = st.radio(
        "Choose option:",
        ["Create New Working Directory", "Load Existing Working Directory"],
    )
    match mode:
        case "Create New Working Directory":
            base_dir = st.text_input("Enter base path:", os.path.expanduser("~"))
            new_dir_name = st.text_input(
                "New working directory name:", "working_output"
            )
            working_dir = os.path.join(base_dir, new_dir_name)

            if st.button("Create Working Directory"):
                if os.path.exists(working_dir):
                    shutil.rmtree(working_dir)
                os.makedirs(working_dir)
                st.session_state["working_dir"] = working_dir
                st.success(f"Created working directory: {os.path.abspath(working_dir)}")

        case "Load Existing Working Directory":
            existing_dir = st.text_input("Enter path to existing directory:")
            if st.button("Load Working Directory"):
                if os.path.exists(existing_dir) and os.path.isdir(existing_dir):
                    st.session_state["working_dir"] = existing_dir
                    st.success(
                        f"Loaded working directory: {os.path.abspath(existing_dir)}"
                    )
                else:
                    st.error("Invalid path.")

    if "working_dir" in st.session_state:
        if uploaded_files := st.file_uploader(
            "Upload image stack",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
        ):
            image_list = [np.array(Image.open(f).convert("L")) for f in uploaded_files]
            image_stack = np.stack(image_list, axis=0)
            Cropper(image_stack, st.session_state["working_dir"]).run()
    else:
        st.info("Please create or load a working directory first.")
