import io
import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from paths.image_paths import CROPPED_IMAGES, RAW_DATA
from visualization.read_image_stack import load_images_from_folder


# Cache image bytes based on image id to speed up repeated display
def pil_from_array(arr: np.ndarray) -> Image.Image:
    # Normalize array to uint8
    if arr.dtype != np.uint8:
        ptp = arr.ptp()
        arr = (
            ((arr - arr.min()) / ptp * 255).astype(np.uint8)
            if ptp
            else np.zeros_like(arr, dtype=np.uint8)
        )
    return Image.fromarray(arr)


@st.cache_data(show_spinner=False, hash_funcs={Image.Image: id})
def get_slice_bytes(img):
    if isinstance(img, np.ndarray):
        pil_img = pil_from_array(img)
    elif isinstance(img, Image.Image):
        pil_img = img
    elif isinstance(img, str):
        pil_img = Image.open(img)
    else:
        pil_img = pil_from_array(np.array(img))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return buf.getvalue()


class ImageViewer:
    def __init__(self, stack):
        processed = []
        for item in stack:
            if isinstance(item, str):
                img = Image.open(item)
            elif isinstance(item, np.ndarray):
                img = pil_from_array(item)
            elif isinstance(item, Image.Image):
                img = item
            else:
                img = pil_from_array(np.array(item))
            processed.append(img)
        self.stack = processed

    def compute_volume_pixels(self):
        data = []
        for img in self.stack:
            arr = np.array(img)
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            data.append(arr.flatten())
        return np.concatenate(data)

    def display_viewer(self):
        st.title("3D Stack Viewer")
        max_idx = len(self.stack) - 1
        idx = st.slider("Slice", 0, max_idx, 0)
        jump = st.text_input(f"Jump to slice (0-{max_idx})", "")
        if jump.strip():
            try:
                j = int(jump)
                if 0 <= j <= max_idx:
                    idx = j
                else:
                    st.warning(f"Number must be between 0 and {max_idx}.")
            except ValueError:
                st.warning("Enter a valid integer.")
        st.image(get_slice_bytes(self.stack[idx]), use_container_width=True, clamp=True)
        pixels = self.compute_volume_pixels()
        arr = np.array(self.stack[idx])
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Volume Histogram")
            fig, ax = plt.subplots()
            ax.hist(pixels, bins=256)
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        with c2:
            st.subheader("Slice Histogram")
            fig2, ax2 = plt.subplots()
            ax2.hist(arr.flatten(), bins=256)
            ax2.set_xlabel("Intensity")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)


class CroppingTool(ImageViewer):
    def display_crop_page(self):
        st.title("Cropping Tool")
        max_idx = len(self.stack) - 1
        slice_idx = st.slider("Current Slice", 0, max_idx, 0, key="crop_slice")
        img = self.stack[slice_idx]
        w, h = img.size
        st.write("\nDraw a rectangle on the image:")
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=2,
            background_image=np.array(img),
            update_streamlit=True,
            height=h,
            width=w,
            drawing_mode="rect",
            key="canvas",
        )
        st.sidebar.header("Crop Settings")
        start_slice = st.sidebar.number_input("Start Slice", 0, max_idx, 0)
        end_slice = st.sidebar.number_input("End Slice", 0, max_idx, max_idx)
        crop_name = st.sidebar.selectbox("Crop Name", list(CROPPED_PATHS.keys()))
        if st.sidebar.button("Create Crop"):
            if not canvas_result.json_data or not canvas_result.json_data.get(
                "objects"
            ):
                st.warning("No rectangle drawn.")
            else:
                obj = canvas_result.json_data["objects"][0]
                left, top = int(obj["left"]), int(obj["top"])
                width, height = int(obj["width"]), int(obj["height"])
                dest_dir = CROPPED_PATHS[crop_name]
                os.makedirs(dest_dir, exist_ok=True)
                for i in range(start_slice, end_slice + 1):
                    slice_img = self.stack[i]
                    cropped = slice_img.crop((left, top, left + width, top + height))
                    out_path = os.path.join(dest_dir, f"{crop_name}_{i}.png")
                    cropped.save(out_path)
                st.success(
                    f"Saved crops for slices {start_slice}-{end_slice} in '{dest_dir}'."
                )


def main():
    images = load_images_from_folder(RAW_DATA, format=".tif")
    viewer = CroppingTool(images)
    page = st.sidebar.radio("Select Page", ["Viewer", "Cropping Tool"])
    if page == "Viewer":
        viewer.display_viewer()
    else:
        viewer.display_crop_page()


if __name__ == "__main__":
    main()
