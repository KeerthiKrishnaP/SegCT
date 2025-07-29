import base64
import importlib
import io
from typing import Any, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from paths.image_paths import RAW_DATA
from visualization.read_image_stack import load_images_from_folder

# ─── PATCH: add missing image_to_url to streamlit.elements.image ──────────────────
# Monkey‑patch so st_canvas can consume PIL Images via data URI
_st_image_mod = importlib.import_module("streamlit.elements.image")


def _image_to_url(image: Image.Image, *args: Any, **kwargs: Any) -> str:
    """
    Convert a PIL Image into a data URI so st_canvas can consume it.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# Overwrite (or add) the function on the module
_st_image_mod.image_to_url = _image_to_url  # type: ignore


class StackCropperApp:
    # Instance attributes with type annotations
    image_stack: np.ndarray  # 3D array of shape (Z, H, W)
    canvas_key: str  # Key for the Streamlit canvas instance

    def __init__(self, image_stack: np.ndarray) -> None:
        """
        Initialize the annotator with a 3D image stack and a canvas key.

        Args:
            image_stack: np.ndarray of shape (Z, H, W)
            canvas_key: Unique key for Streamlit canvas
        """
        self.image_stack = image_stack
        self.canvas_key = "rect_canvas"

        # initialize drawing state in Streamlit session
        if "drawing" not in st.session_state:
            st.session_state.drawing = False  # type: bool

    def select_slice(self) -> int:
        """Let the user pick which Z-slice to annotate."""
        max_index: int = self.image_stack.shape[0] - 1
        return st.slider("Select slice for annotation", 0, max_index, 0)

    def draw_canvas(self, image: np.ndarray) -> Any:
        """Show the canvas for drawing rectangles on the given 2D image."""
        bg_img: Image.Image = Image.fromarray(image)
        return st_canvas(
            fill_color="rgba(0, 0, 255, 0.2)",  # RGBA fill for rectangles
            stroke_width=2,
            background_image=bg_img,  # type: ignore
            height=image.shape[0],
            width=image.shape[1],
            drawing_mode="rect",
            key=self.canvas_key,
        )

    def show_cropped_subimages(
        self, rect: Tuple[int, int, int, int], z_start: int, z_end: int
    ) -> None:
        """Crop the volume between z_start and z_end and display sub-images."""
        x, y, w, h = rect
        substack: np.ndarray = self.image_stack[
            z_start : z_end + 1, y : y + h, x : x + w
        ]
        st.subheader(f"Cropped sub-images (slices {z_start}–{z_end})")
        pil_list = [Image.fromarray(s) for s in substack]
        st.image(pil_list, width=200, clamp=True)

    def run(self) -> None:
        """Main entrypoint: UI controls for cropping options, annotation, and display."""
        st.title("3D Rectangular Crop Annotator")

        # --- Cropping slice range input ---
        z_min, z_max = 0, self.image_stack.shape[0] - 1
        z_start, z_end = st.slider(
            "Select slice range for cropping", z_min, z_max, (z_min, z_max)
        )

        # --- Annotation slice selector ---
        slice_idx = self.select_slice()
        st.subheader(f"Annotating slice {slice_idx}")
        img = self.image_stack[slice_idx]

        # Start drawing button
        if not st.session_state.drawing:
            if st.button("Start Drawing Rectangle"):
                st.session_state.drawing = True

        # Drawing canvas and capture
        if st.session_state.drawing:
            canvas_result = self.draw_canvas(img)

            if st.button("End Drawing Rectangle"):
                data = canvas_result.json_data  # type: ignore
                rect = (0, 0, 0, 0)
                if data and "objects" in data:
                    for obj in data["objects"]:
                        if obj.get("type") == "rect":
                            x = int(obj["left"])
                            y = int(obj["top"])
                            w = int(obj["width"])
                            h = int(obj["height"])
                            rect = (x, y, w, h)
                            break
                st.session_state.drawing = False
                self.show_cropped_subimages(rect, z_start, z_end)
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()


def main() -> None:
    """Load the image stack and launch the annotator."""
    images = load_images_from_folder(RAW_DATA, format=".tif")
    viewer = StackCropperApp(images)
    viewer.run()


if __name__ == "__main__":
    main()
