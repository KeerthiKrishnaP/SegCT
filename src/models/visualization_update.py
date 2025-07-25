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


import os

import numpy as np
import streamlit as st
import tifffile
from PIL import Image
from streamlit_drawable_canvas import st_canvas


class StackCropperApp:
    """
    Streamlit app for cropping regions from a 3D TIFF image stack.
    """

    # Mapping of crop categories to output folders
    CATEGORY_FOLDERS = {
        "Cropped Image": os.path.join("crops", "cropped_images"),
        "Warp Yarn": os.path.join("crops", "warp_yarn"),
        "Weft Yarn": os.path.join("crops", "weft_yarn"),
        "Void": os.path.join("crops", "void"),
    }

    def __init__(self):
        # Ensure output directories exist
        for folder in self.CATEGORY_FOLDERS.values():
            os.makedirs(folder, exist_ok=True)

    def load_stack(self, filepath: str) -> np.ndarray:
        """
        Load a TIFF stack from the given file path.
        """
        return tifffile.imread(filepath)

    def save_crop(
        self,
        stack: np.ndarray,
        bbox: tuple,
        z_start: int,
        z_end: int,
        category: str,
        base_name: str = "crop",
    ):
        """
        Save cropped slices to the folder matching the category.
        """
        x, y, w, h = bbox
        output_dir = self.CATEGORY_FOLDERS[category]

        for z in range(z_start, z_end + 1):
            slice_img = stack[z]
            cropped = slice_img[y : y + h, x : x + w]
            filename = f"{base_name}_{category.replace(' ', '_')}_z{z}.png"
            out_path = os.path.join(output_dir, filename)
            Image.fromarray(cropped).save(out_path)

    def run(self):
        """
        Launch the Streamlit interface.
        """
        st.set_page_config(page_title="3D Stack Cropper", layout="wide")
        st.title("3D Image Stack Cropper")

        # Input: path to the TIFF stack
        filepath = st.text_input("Enter path to 3D TIFF stack")

        if not filepath:
            st.info("Please provide the path to a TIFF stack to begin.")
            return

        try:
            stack = self.load_stack(filepath)
        except Exception as e:
            st.error(f"Failed to load stack: {e}")
            return

        depth, height, width = stack.shape

        # Sidebar controls
        st.sidebar.header("Slice Navigation")
        slice_idx = st.sidebar.slider("Select Slice", 0, depth - 1, 0)

        st.sidebar.header("Crop Settings")
        z_start = st.sidebar.number_input(
            "Start Slice", min_value=0, max_value=depth - 1, value=0, step=1
        )
        z_end = st.sidebar.number_input(
            "End Slice", min_value=0, max_value=depth - 1, value=depth - 1, step=1
        )
        crop_name = st.sidebar.selectbox(
            "Crop Name", list(self.CATEGORY_FOLDERS.keys())
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader(f"Slice {slice_idx} of {depth - 1}")
            img = stack[slice_idx]
            img_pil = Image.fromarray(img)

            # Cropping mode toggles
            if st.button("Start Cropping"):
                st.session_state.mode = "draw"
            if st.button("End Cropping"):
                st.session_state.mode = "view"

            canvas = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=img_pil,
                update_streamlit=True,
                drawing_mode=(
                    "rectangle"
                    if st.session_state.get("mode") == "draw"
                    else "transform"
                ),
                height=height,
                width=width,
                key="canvas",
            )

        with col2:
            st.subheader("Create Crop")

            if canvas.json_data and canvas.json_data.get("objects"):
                # Use the first rectangle drawn
                obj = canvas.json_data["objects"][0]
                left = int(obj["left"])
                top = int(obj["top"])
                w_box = int(obj["width"])
                h_box = int(obj["height"])

                st.write(f"BBox: x={left}, y={top}, w={w_box}, h={h_box}")

                if st.button("Create Crop"):
                    base = os.path.splitext(os.path.basename(filepath))[0]
                    self.save_crop(
                        stack,
                        (left, top, w_box, h_box),
                        int(z_start),
                        int(z_end),
                        crop_name,
                        base_name=base,
                    )
                    st.success(f"Saved '{crop_name}' from slices {z_start} to {z_end}.")
            else:
                st.info(
                    "Draw a rectangle on the image and click 'End Cropping' to enable saving."
                )


def main():
    images = load_images_from_folder(RAW_DATA, format=".tif")
    viewer = StackCropperApp(images)
    page = st.sidebar.radio("Select Page", ["Viewer", "Cropping Tool"])
    if page == "Viewer":
        viewer.display_viewer()
    else:
        viewer.display_crop_page()


if __name__ == "__main__":
    main()
