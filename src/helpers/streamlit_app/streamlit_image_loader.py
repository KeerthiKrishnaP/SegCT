import base64
import importlib
import io
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import streamlit as st
from numpy.typing import NDArray
from PIL import Image
from streamlit_drawable_canvas import CanvasResult, st_canvas

# ───────────────────────────────────────────
# Patch: use image_to_url for canvas background
_st_image_mod = importlib.import_module("streamlit.elements.image")


def _image_to_url(image, format: str | None = None, *args, **kwargs) -> str:
    """Convert a PIL image to a base64-encoded data URL for Streamlit canvas."""

    # --- Defensive handling ---
    if not isinstance(format, str):
        format = getattr(image, "format", "PNG") or "PNG"

    fmt = format.lower().lstrip(".")  # type: ignore
    buffered = io.BytesIO()
    image.save(buffered, format=fmt.upper())  # PIL expects "PNG", "TIFF", etc.
    b64 = base64.b64encode(buffered.getvalue()).decode("ascii")

    match fmt:
        case "png":
            mime = "image/png"
        case "tiff" | "tif":
            mime = "image/tiff"
        case _:
            mime = f"image/{fmt}"

    return f"data:{mime};base64,{b64}"


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


def save_stack_to_h5(stack: NDArray, filepath: str):
    """
    Save a 3D NumPy array (n_images, height, width) to an HDF5 (.h5) file.

    Parameters
    ----------
    stack : np.ndarray
        3D NumPy array containing the image stack.
    filepath : str
        Path to save the .h5 file.
    dataset_name : str, optional
        Name of the dataset inside the HDF5 file (default is 'stack').
    """
    if not isinstance(stack, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    if stack.ndim != 3:
        raise ValueError("Input array must be 3D (n_images, height, width).")

    if not os.path.exists(filepath):
        Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)
        dataset_name = "stack"

    with h5py.File(filepath, "w") as file:
        file.create_dataset(dataset_name, data=stack, compression="gzip")
        file.attrs["shape"] = stack.shape
        file.attrs["dtype"] = str(stack.dtype)

    print(f"✅ Saved stack with shape {stack.shape} to {filepath}")


def load_stack_from_h5(filepath: str) -> NDArray:
    """
    Load a 3D NumPy array from an HDF5 (.h5) file.

    Parameters
    ----------
    filepath : str
        Path to the .h5 file.
    Returns
    -------
    np.ndarray
        3D NumPy array (n_images, height, width).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} does not exist.")

    with h5py.File(filepath, "r") as file:
        print("Keys:", list(file.keys()))
        dataset_name = list(file.keys())[0]
        data = np.array(file[dataset_name])

    print(f"✅ Loaded stack with shape {data.shape} from {filepath}")

    return data


def crop_3d_stack(
    volume: NDArray,
    rectangle_cor: tuple[int, int, int, int],
    end_slice: int,
    start_slice: int = 0,
) -> NDArray:
    """
    Crops a rectangular region from each 2D slice in a 3D numpy array.

    Parameters
    ----------
    volume : np.ndarray
        3D array of shape (num_slices, height, width)
    start_slice : int
        Index of the first slice to include (inclusive)
    end_slice : int
        Index of the last slice to include (exclusive)
    x_start, x_end : int
        X-coordinate range (columns)
    y_start, y_end : int
        Y-coordinate range (rows)

    Returns
    -------
    cropped_volume : np.ndarray
        Cropped 3D array
    """
    x_start, y_start, w, h = rectangle_cor
    x_end = x_start + w
    y_end = y_start + h

    return volume[start_slice:end_slice, y_start:y_end, x_start:x_end]


def load_structural_tensor_images(results_dir: str) -> Dict[str, NDArray]:
    """Load S11...S23 images for a given slice if they exist."""
    components = ["S11", "S22", "S33", "S12", "S13", "S23"]
    images = {}
    for component in components:
        comp_dir = os.path.join(results_dir, component)
        if os.path.exists(comp_dir):
            file_path = os.path.join(comp_dir, f"{component}.h5")
            if os.path.exists(file_path):
                images[component] = load_stack_from_h5(file_path)

    return images


def normalize_stack(
    data: NDArray | dict[str, NDArray],
) -> NDArray | Dict[str, NDArray]:
    """
    Normalize image data (2D or 3D) or all stacks in a dictionary to 0–255 uint8.

    Parameters
    ----------
    data : np.ndarray or dict[str, np.ndarray]
        Input image stack or dictionary of stacks.

    Returns
    -------
    np.ndarray or dict[str, np.ndarray]
        Normalized image stack(s) as uint8.
    """

    def _normalize_single(stack: np.ndarray) -> np.ndarray:
        stack_min = stack.min()
        stack_max = stack.max()
        print(f"Normalizing stack with min {stack_min}, max {stack_max}")
        normalized = (stack - stack_min) / (stack_max - stack_min)

        return (normalized * 255).astype(np.uint8)

    # --- Handle dictionary input ---
    if isinstance(data, dict):
        return {name: _normalize_single(stack) for name, stack in data.items()}

    # --- Handle single array input ---
    elif isinstance(data, np.ndarray):
        return _normalize_single(data)

    else:
        raise TypeError(
            "Input must be either a NumPy array or a dict[str, np.ndarray]."
        )
