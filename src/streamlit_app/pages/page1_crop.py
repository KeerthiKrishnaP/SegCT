import os
import shutil

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from streamlit_app.utils.image_utilis import resize_if_needed


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

    def draw_crop_rect(self, image: np.ndarray):
        img_pil = Image.fromarray(image)
        resized_img, scale = resize_if_needed(img_pil)
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",
            stroke_width=2,
            background_image=resized_img,
            height=resized_img.size[1],
            width=resized_img.size[0],
            drawing_mode="rect",
            key=f"{self.prefix}_canvas",
        )
        return canvas_result, scale, img_pil.size

    def crop_all(self, rect_coords, start_idx, end_idx):
        x, y, w, h = rect_coords
        for i in range(start_idx, end_idx + 1):
            img = Image.fromarray(self.image_stack[i])
            cropped = img.crop((x, y, x + w, y + h))
            filename = f"cropped_slice{i}.tiff"
            save_path = os.path.join(self.crop_dir, filename)
            cropped.save(save_path)
        st.success(f"Saved cropped images {start_idx} → {end_idx} to {self.crop_dir}")

    def run(self):
        slice_idx = self.select_slice()
        img = self.image_stack[slice_idx]
        canvas_result, scale, orig_size = self.draw_crop_rect(img)

        # Pick slice range for cropping
        st.write("### Slice Range for Cropping")
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

        if st.button("Crop All Images", key=f"{self.prefix}_crop_button"):
            data = canvas_result.json_data
            if data and "objects" in data and len(data["objects"]) > 0:
                rect = data["objects"][0]  # take first rect
                # Map rect coords back to original resolution
                x = int(rect["left"] / scale)
                y = int(rect["top"] / scale)
                w = int(rect["width"] / scale)
                h = int(rect["height"] / scale)
                self.crop_all((x, y, w, h), int(start_idx), int(end_idx))
            else:
                st.warning("Please draw a rectangle before cropping.")


def page1_crop_images():
    st.header("Load & Crop Images")

    # --- Choose mode: create or load ---
    mode = st.radio(
        "Choose option:",
        ["Create New Working Directory", "Load Existing Working Directory"],
        key="page1_mode_radio",
    )

    working_dir = None

    if mode == "Create New Working Directory":
        base_dir = st.text_input(
            "Enter base path:",
            os.path.expanduser("~"),
            key="page1_base_dir_input",
        )

        new_dir_name = st.text_input(
            "Enter a name for the working directory:",
            "working_output",
            key="page1_new_dir_name_input",
        )
        working_dir = os.path.join(base_dir, new_dir_name)
        if os.path.exists(working_dir):
            st.warning("Directory already exists. Its being overwritten.")
        if st.button("Create Working Directory", key="page1_create_dir_button"):
            if os.path.exists(working_dir) and os.path.isdir(working_dir):
                shutil.rmtree(working_dir)
            os.makedirs(working_dir)
            st.session_state["working_dir"] = working_dir
            st.success(
                f"Created working directory: {os.path.abspath(working_dir)}", icon="📁"
            )

    elif mode == "Load Existing Working Directory":
        existing_dir = st.text_input(
            "Enter path to existing working directory:",
            key="page1_existing_dir_input",
        )

        if st.button("Load Working Directory", key="page1_load_dir_button"):
            if os.path.exists(existing_dir) and os.path.isdir(existing_dir):
                working_dir = existing_dir
                st.session_state["working_dir"] = working_dir
                st.success(
                    f"Loaded working directory: {os.path.abspath(working_dir)}",
                    icon="📂",
                )
            else:
                st.error("❌ The path does not exist or is not a directory.")

    if "working_dir" in st.session_state:
        uploaded_files = st.file_uploader(
            "Upload image stack",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
            key="page1_file_uploader",
        )

        if uploaded_files:
            image_list = [np.array(Image.open(f).convert("L")) for f in uploaded_files]
            image_stack = np.stack(image_list, axis=0)
            cropper = Cropper(
                image_stack, st.session_state["working_dir"], prefix="page1_cropper"
            )
            cropper.run()
    else:
        st.info("Please create or load a working directory first.", icon="ℹ️")
