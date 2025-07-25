import base64
import io

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from paths.image_paths import CROPPED_IMAGES, RAW_DATA
from visualization.read_image_stack import load_images_from_folder


class CreaterFeatureDataSets:
    """
    Streamlit app class for loading a 3D NumPy image stack and displaying slices
    with ROI cropping tool functionality.
    """

    def __init__(self, stack: np.ndarray, downsample_factor: int = 2) -> None:
        """
        Initialize with a pre-loaded 3D NumPy image stack.

        Parameters:
            stack (np.ndarray): 3D array of shape (slices, height, width) or (slices, height, width, channels).
            downsample_factor (int): Factor by which to downsample for faster preview.
        """
        self.stack = stack
        self.downsample_factor = downsample_factor

    @staticmethod
    @st.cache_data
    def load_image_stack(array: np.ndarray) -> np.ndarray:
        """
        Cache and return a 3D NumPy image stack.
        """
        return array

    @staticmethod
    @st.cache_data
    def get_thumbnail_slice(array: np.ndarray, idx: int, factor: int) -> np.ndarray:
        """
        Return a single downsampled slice, cached per index.
        """
        slice_img = array[idx]
        if slice_img.ndim == 2:
            return slice_img[::factor, ::factor]
        else:
            return slice_img[::factor, ::factor, :]

    @staticmethod
    @st.cache_data
    def get_full_slice(array: np.ndarray, idx: int) -> np.ndarray:
        """
        Return the full-resolution slice, cached per index.
        """
        return array[idx]

    @staticmethod
    def pil_to_base64_url(pil_img: Image.Image) -> str:
        """
        Convert a PIL Image to a base64-encoded data URL.
        """
        buff = io.BytesIO()
        pil_img.save(buff, format="PNG")
        encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def display_slice_viewer(self) -> None:
        """
        Display a Streamlit slider to view slices, and an ROI cropping tool
        with Start Draw / End Draw buttons and rectangle annotations.
        """
        num_slices = self.stack.shape[0]

        # Initialize session state
        if "slice_idx" not in st.session_state:
            st.session_state.slice_idx = 0
        if "drawing" not in st.session_state:
            st.session_state.drawing = False
        if "rois" not in st.session_state:
            st.session_state.rois = {}

        # Slider control
        slice_idx = st.slider(
            label="Select slice index",
            min_value=0,
            max_value=num_slices - 1,
            value=st.session_state.slice_idx,
            step=1,
        )
        st.session_state.slice_idx = slice_idx

        # Retrieve thumbnail image for current slice and convert to PIL
        thumb_img = self.get_thumbnail_slice(
            self.stack, slice_idx, self.downsample_factor
        )
        if thumb_img.dtype != np.uint8:
            thumb_norm = 255 * (thumb_img - thumb_img.min()) / (thumb_img.ptp() or 1)
            thumb_uint8 = thumb_norm.astype(np.uint8)
        else:
            thumb_uint8 = thumb_img
        pil_thumb = Image.fromarray(thumb_uint8)

        st.image(
            pil_thumb,
            caption=f"Slice {slice_idx + 1}/{num_slices} (preview)",
            use_container_width=True,
        )

        # ROI Cropping Tool Controls
        st.markdown("---")
        st.markdown("### ROI Cropping Tool")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Draw"):
                st.session_state.drawing = True
        with col2:
            if st.button("End Draw"):
                st.session_state.drawing = False

        # Prepare background image URL
        bg_url = self.pil_to_base64_url(pil_thumb)

        # Canvas for drawing rectangles
        canvas_result = st_canvas(
            stroke_width=2,
            stroke_color="#FF0000",
            background_image_url=bg_url,
            update_streamlit=True,
            height=pil_thumb.height,
            width=pil_thumb.width,
            drawing_mode="rect" if st.session_state.drawing else "transform",
            key=f"canvas_{slice_idx}",
        )

        # On End Draw, capture rectangles
        if not st.session_state.drawing and canvas_result.json_data:
            objects = canvas_result.json_data.get("objects", [])
            rois = st.session_state.rois.setdefault(slice_idx, [])
            for obj in objects:
                if obj.get("type") == "rect":
                    left = obj.get("left", 0)
                    top = obj.get("top", 0)
                    width = obj.get("width", 0)
                    height = obj.get("height", 0)
                    rois.append((left, top, width, height))
            # Clear canvas for next draw session
            canvas_result.json_data["objects"] = []

        # Display stored ROIs for this slice
        if slice_idx in st.session_state.rois and st.session_state.rois[slice_idx]:
            st.markdown("**Saved ROIs:**")
            for i, (x, y, w, h) in enumerate(st.session_state.rois[slice_idx], start=1):
                st.write(f"ROI {i}: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")

    def run(self) -> None:
        """
        Run the Streamlit app: load images and display the viewer with ROI tool.
        """
        st.title("3D Image Stack ROI Annotator")
        if isinstance(self.stack, np.ndarray) and self.stack.size:
            _ = self.load_image_stack(self.stack)
            st.markdown("**Step 2:** Scroll through image slices.")
            self.display_slice_viewer()
        else:
            st.error("No valid image stack provided.")


if __name__ == "__main__":
    images = load_images_from_folder(RAW_DATA, format=".tif")
    app = CreaterFeatureDataSets(images)
    app.run()
