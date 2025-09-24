import base64
import importlib
import io
import os
from typing import Tuple

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import CanvasResult, st_canvas

# ───────────────────────────────────────────
# Patch: use image_to_url for canvas background
_st_image_mod = importlib.import_module("streamlit.elements.image")


def _image_to_url(image, *args, **kwargs):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


_st_image_mod.image_to_url = _image_to_url
# ───────────────────────────────────────────


def _resize_if_needed(
    img: Image.Image, max_dim: int = 800
) -> Tuple[Image.Image, float]:
    """Resize image to fit within max_dim, return resized + scale factors."""
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        return img.resize(new_size, Image.LANCZOS), scale
    return img, 1.0


def load_images_from_dir(dir: str, ext: str = ".tiff") -> np.ndarray | None:
    if not os.path.exists(dir):
        return None
    files = sorted([f for f in os.listdir(dir) if f.endswith(ext)])
    if not files:
        return None
    image_list = [
        np.array(Image.open(os.path.join(dir, f)).convert("L")) for f in files
    ]
    return np.stack(image_list, axis=0)


def draw_rectangle_canvas(
    image,
    prefix: str,
    fill_color="rgba(255, 0, 0, 0.2)",
    stroke_width=2,
) -> Tuple[CanvasResult, float, tuple[int, int]]:
    """
    Display a resizable image on a canvas for rectangle drawing.

    Returns:
        canvas_result: Streamlit canvas object
        scale: scaling factor used for resizing
        original_size: (width, height) of the original image
    """
    img_pil = Image.fromarray(image)
    resized_img, scale = _resize_if_needed(img_pil)

    canvas_result = st_canvas(
        fill_color=fill_color,
        stroke_width=stroke_width,
        background_image=resized_img,  # type: ignore
        height=resized_img.size[1],
        width=resized_img.size[0],
        drawing_mode="rect",
        key=f"{prefix}_canvas",
    )
    return canvas_result, scale, img_pil.size


def show_component_images(images: dict, slice_idx: int) -> None:
    """
    Show structural tensor component images in a grid of columns.
    """
    if not images:
        st.info("No images found for selected slice.")
        return

    cols = st.columns(len(images))
    for i, (comp, img) in enumerate(images.items()):
        with cols[i]:
            st.image(
                img, caption=f"{comp} - slice {slice_idx}", use_container_width=True
            )
