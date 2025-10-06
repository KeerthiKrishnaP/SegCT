import base64
import importlib
import io
import os
from collections import defaultdict
from typing import Any, Tuple

import numpy as np
import streamlit as st
from numpy.typing import NDArray
from PIL import Image
from streamlit_drawable_canvas import CanvasResult, st_canvas

# ───────────────────────────────────────────
# Patch: use image_to_url for canvas background
_st_image_mod = importlib.import_module("streamlit.elements.image")


def _image_to_url(image, format, *args, **kwargs):
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
    match format:
        case "PNG":
            return f"data:image/png;base64,{b64}"
        case "TIFF":
            return f"data:image/tiff;base64,{b64}"
        case _:
            raise ValueError("Define the right format of the image")


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


def slice_viewer(
    image_stack: np.ndarray, prefix="slice_viewer", caption="Select slice"
) -> Tuple[int, np.ndarray[Any, Any]]:
    num_slices = image_stack.shape[0]
    idx = st.slider(
        caption,
        min_value=0,
        max_value=num_slices - 1,
        value=0,
        key=f"{prefix}_slice_slider",
    )
    return idx, image_stack[idx]


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
                np.clip(img, 0, 255).astype(np.uint8),
                caption=f"{comp} - slice {slice_idx}",
                use_container_width=False,
            )


def load_structural_tensor_dict_from_images(
    results_dir,
) -> dict[str, NDArray]:
    """Load S11...S23 images for a given slice if they exist."""
    comps = ["S11", "S22", "S33", "S12", "S13", "S23"]
    components_dict = defaultdict()

    for comp in comps:
        components_dict[comp] = load_images_from_dir(
            os.path.join(results_dir, comp), ext=".tiff"
        )

    return components_dict
