import os
import shutil
from typing import Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

from helpers.streamlit_app.streamlit_image_loader import (
    crop_3d_stack,
    draw_rectangle_canvas,
    save_stack_to_h5,
    slice_viewer,
)

# ============================================================
# Utility Functions
# ============================================================


def create_working_directory(base_path: str, dir_name: str) -> str:
    """Create or recreate a clean working directory."""
    working_dir = os.path.join(base_path, dir_name)
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)
    return working_dir


def validate_directory(path: str) -> bool:
    """Check if a path is a valid directory."""
    return os.path.exists(path) and os.path.isdir(path)


def load_image_stack(uploaded_files) -> np.ndarray:
    """Convert uploaded image files into a 3D NumPy stack."""
    image_list = [np.array(Image.open(f).convert("L")) for f in uploaded_files]
    return np.stack(image_list, axis=0)


# ============================================================
# Cropper Core Logic
# ============================================================


class Cropper:
    """Handles interactive 3D image stack cropping and saving."""

    def __init__(
        self, image_stack: np.ndarray, save_dir: str, prefix: str = "page1_cropper"
    ) -> None:
        self.image_stack = image_stack
        self.save_dir = save_dir
        self.prefix = prefix
        self.crop_dir = os.path.join(save_dir, "crops", "data_stack")
        os.makedirs(self.crop_dir, exist_ok=True)

    # ---------------------
    # Public Interface
    # ---------------------
    def run(self) -> None:
        """Launch interactive cropping interface."""
        _, img = slice_viewer(self.image_stack, prefix=self.prefix)

        canvas_result, scale, _ = draw_rectangle_canvas(
            img, prefix=self.prefix, fill_color="rgba(255, 0, 0, 0.2)"
        )

        st.write("### Slice Range for Cropping")
        start_idx, end_idx, file_name = self._input_crop_settings()
        crop_button = st.button("Crop All Images", key=f"{self.prefix}_crop_button")

        if crop_button:
            self._handle_crop(canvas_result, scale, start_idx, end_idx, file_name)

    # ---------------------
    # Internal Helpers
    # ---------------------
    def _input_crop_settings(self) -> Tuple[int, int, str]:
        """Render Streamlit inputs for crop settings."""
        start_idx = st.number_input(
            "Start slice index", 0, self.image_stack.shape[0] - 1, 0
        )
        end_idx = st.number_input(
            "End slice index",
            0,
            self.image_stack.shape[0] - 1,
            self.image_stack.shape[0] - 1,
        )
        file_name = st.text_input("File name for cropped stack:", "cropped_stack")
        return int(start_idx), int(end_idx), file_name

    def _handle_crop(
        self, canvas_result, scale: float, start_idx: int, end_idx: int, file_name: str
    ) -> None:
        """Handle rectangle drawing and initiate cropping."""
        data = canvas_result.json_data
        if not (data and "objects" in data and len(data["objects"]) > 0):
            st.warning("Please draw a rectangle before cropping.")
            return

        rect = data["objects"][0]
        coords = self._extract_rectangle_coords(rect, scale)
        self._crop_and_save(coords, start_idx, end_idx, file_name)

    def _extract_rectangle_coords(
        self, rect_data: dict, scale: float
    ) -> Tuple[int, int, int, int]:
        """Convert rectangle JSON data to pixel coordinates."""
        x = int(rect_data["left"] / scale)
        y = int(rect_data["top"] / scale)
        w = int(rect_data["width"] / scale)
        h = int(rect_data["height"] / scale)
        return x, y, w, h

    def _crop_and_save(
        self,
        rect_coords: Tuple[int, int, int, int],
        start_idx: int,
        end_idx: int,
        file_name: str,
    ) -> None:
        """Crop and save the 3D image stack to an HDF5 file."""
        cropped_image = crop_3d_stack(
            volume=self.image_stack,
            rectangle_cor=rect_coords,
            start_slice=start_idx,
            end_slice=end_idx,
        )
        output_path = os.path.join(self.crop_dir, f"{file_name}.h5")
        save_stack_to_h5(cropped_image, output_path)
        st.success(f"✅ Saved cropped stack ({start_idx} → {end_idx}) to {output_path}")


# ============================================================
# Streamlit App Entry Point
# ============================================================


def app() -> None:
    """Main Streamlit app for image stack cropping."""
    st.header("🖼️ Load & Crop Image Stacks")

    working_dir = _select_or_create_working_dir()
    if not working_dir:
        st.info("Please create or load a working directory first.")
        return

    if uploaded_files := st.file_uploader(
        "Upload image stack",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    ):
        image_stack = load_image_stack(uploaded_files)
        Cropper(image_stack, working_dir).run()


def _select_or_create_working_dir() -> Optional[str]:
    """Handle working directory creation/loading."""
    mode = st.radio(
        "Choose option:",
        ["Create New Working Directory", "Load Existing Working Directory"],
    )

    if mode == "Create New Working Directory":
        base_dir = st.text_input("Enter base path:", os.path.expanduser("~"))
        new_dir_name = st.text_input("New working directory name:", "working_output")
        if st.button("Create Working Directory"):
            working_dir = create_working_directory(base_dir, new_dir_name)
            st.session_state["working_dir"] = working_dir
            st.success(f"Created working directory: {os.path.abspath(working_dir)}")

    elif mode == "Load Existing Working Directory":
        existing_dir = st.text_input("Enter path to existing directory:")
        if st.button("Load Working Directory"):
            if validate_directory(existing_dir):
                st.session_state["working_dir"] = existing_dir
                st.success(f"Loaded working directory: {os.path.abspath(existing_dir)}")
            else:
                st.error("Invalid path.")

    return st.session_state.get("working_dir")
