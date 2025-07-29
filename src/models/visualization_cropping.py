import base64
import io
import os
from glob import glob
import importlib

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

from paths.image_paths import CROPPED_IMAGES, RAW_DATA
from visualization.read_image_stack import load_images_from_folder

from collections import defaultdict


_st_image_mod = importlib.import_module("streamlit.elements.image")

# Patch to use the image ot url
def _image_to_url(image, *args, **kwargs):
    """
    Convert a PIL Image into a data URI so st_canvas can consume it.
    Ignores any additional parameters.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# overwrite (or add) the function on the module
_st_image_mod.image_to_url = _image_to_url

# ───────────────────────────────────────────

class IinteractiveCropping:
    """
    Streamlit application for browsing a 3D image stack,
    specifying rectangular ROIs via numeric inputs,
    previewing each ROI, and batch-exporting cropped patches.
    """

    def __init__(self, stack: np.ndarray, canvas_key: str = "rect_canvas"):
        """Initialize with a 3D (or 4D with channels) NumPy array."""
        if stack.ndim not in [3, 4]:
            raise ValueError("Stack must be a 3D array or 4D with channels.")
        self.stack = stack
        self.depth = stack.shape[0]
        self.canvas_key = canvas_key

        # initialize storage
        if "rectangles" not in st.session_state:
            st.session_state.rectangles = []

    def run(self):
        """Render the Streamlit UI and handle interactions."""
        st.title("3D Stack ROI Crop & Export")

        # Slice navigation
        slice_idx = st.sidebar.slider("Slice index", 0, self.depth - 1, 0)
        st.sidebar.write(f"Displaying slice {slice_idx} of {self.depth - 1}")
        slice_arr = self.stack[slice_idx]

        # Convert to PIL image
        pil_img = (
            Image.fromarray(slice_arr)
            if slice_arr.ndim == 2
            else Image.fromarray(slice_arr)
        )
        width, height = pil_img.size

        # Draw rectangle to get the ROI
    def draw_canvas(self, image: np.ndarray):
        """Show the canvas for drawing rectangles on the given 2D image."""
        # convert numpy array to PIL Image
        return st_canvas(
            fill_color="rgba(0, 0, 255, 0.2)",  # semi-transparent fill
            stroke_width=2,
            background_image=Image.fromarray(image),  # type: ignore
            height=image.shape[0],
            width=image.shape[1],
            drawing_mode="rect",
            key=self.canvas_key,
        )

    def get_coordinates_from_canvas(canvas_result) -> dict:
        coordinates = defaultdict()
        data = canvas_result.json_data


        return
        # ROI input controls
        st.sidebar.header("Define ROI (Preview Below)")
        x = st.sidebar.number_input(
            "X (pixels)", min_value=0, max_value=width - 1, value=int(width * 0.1)
        )
        y = st.sidebar.number_input(
            "Y (pixels)", min_value=0, max_value=height - 1, value=int(height * 0.1)
        )
        w = st.sidebar.number_input(
            "Width (pixels)", min_value=1, max_value=width - x, value=int(width * 0.3)
        )
        h = st.sidebar.number_input(
            "Height (pixels)",
            min_value=1,
            max_value=height - y,
            value=int(height * 0.3),
        )

        # Preview the current ROI on the slice and as a zoomed crop
        disp_with_box = pil_img.copy()
        draw = ImageDraw.Draw(disp_with_box)
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        st.subheader("ROI Preview on Slice")
        st.image(
            disp_with_box, caption=f"Slice {slice_idx} with ROI", use_column_width=True
        )

        # Show cropped sub-image in separate panel
        patch = pil_img.crop((x, y, x + w, y + h))
        st.subheader("Cropped ROI Preview")
        st.image(patch, caption="Cropped region", use_column_width=False)

        # Add ROI button
        if st.sidebar.button("Add ROI"):
            st.session_state.rois.append(
                {"slice": slice_idx, "x": x, "y": y, "w": w, "h": h}
            )

        # Display current ROIs
        st.subheader("Current ROIs")
        if st.session_state.rois:
            for i, roi in enumerate(st.session_state.rois):
                st.write(
                    f"{i}: Slice {roi['slice']}, x={roi['x']}, y={roi['y']}, w={roi['w']}, h={roi['h']}"
                )
            if st.button("Clear All ROIs"):
                st.session_state.rois = []
        else:
            st.write("No ROIs defined yet.")

        # Overlay all ROIs on display image
        disp_all = pil_img.copy()
        draw_all = ImageDraw.Draw(disp_all)
        for roi in st.session_state.rois:
            if roi["slice"] == slice_idx:
                x0, y0 = roi["x"], roi["y"]
                x1, y1 = x0 + roi["w"], y0 + roi["h"]
                draw_all.rectangle([x0, y0, x1, y1], outline="blue", width=2)
        st.subheader("All ROIs on Current Slice")
        st.image(
            disp_all, caption=f"Slice {slice_idx} with all ROIs", use_column_width=True
        )

        # Export controls
        output_dir = st.text_input("Output directory", value="output_crops")
        if st.button("Crop & Save All ROIs"):
            if not st.session_state.rois:
                st.error("No ROIs to save. Define at least one.")
            else:
                count = self.crop_and_save(output_dir)
                st.success(f"Saved {count} cropped images to '{output_dir}'")

    def crop_and_save(self, output_dir: str) -> int:
        """
        Crop all stored ROIs and save to disk.
        Returns total saved count.
        """
        os.makedirs(output_dir, exist_ok=True)
        count = 0
        for i, roi in enumerate(st.session_state.rois):
            idx = roi["slice"]
            arr = self.img_stack[idx]
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            patch_arr = (
                arr[y : y + h, x : x + w]
                if arr.ndim == 2
                else arr[y : y + h, x : x + w, ...]
            )
            img_patch = Image.fromarray(patch_arr)
            fname = os.path.join(output_dir, f"slice_{idx}_roi_{i}.png")
            img_patch.save(fname)
            count += 1
        return count


if __name__ == "__main__":
    # Sidebar: choose input
    stack = load_images_from_folder(RAW_DATA, format=".tif")
    # Run the app
    app = ROIApp(stack)
    app.run()

─────────────────────────────────────


class RectAnnotator:
    def __init__(self, image_stack: np.ndarray, canvas_key: str = "rect_canvas"):
        """
        image_stack: 3D numpy array of shape (Z, H, W)
        canvas_key:   Streamlit key for the drawable canvas
        """
        self.image_stack = image_stack
        self.canvas_key = canvas_key

        # initialize storage
        if "rectangles" not in st.session_state:
            st.session_state.rectangles = []

    def select_slice(self) -> int:
        """Let the user pick which Z-slice to annotate."""
        return st.slider(
            "Select slice",
            min_value=0,
            max_value=self.image_stack.shape[0] - 1,
            value=0,
        )



    def save_rectangles(self, canvas_result, slice_idx: int):
        """Extract any drawn rects and append their coords to session_state."""
        data = canvas_result.json_data
        if not data or "objects" not in data:
            return
        for obj in data["objects"]:
            if obj.get("type") == "rect":
                coords = {
                    "slice": slice_idx,
                    "x": obj["left"],
                    "y": obj["top"],
                    "w": obj["width"],
                    "h": obj["height"],
                }
                st.session_state.rectangles.append(coords)

    def run(self):
        """Main entrypoint: select slice, draw, save on button, and display."""
        slice_idx = self.select_slice()
        img = self.image_stack[slice_idx]
        canvas_result = self.draw_canvas(img)

        if st.button("End Rectangle"):
            self.save_rectangles(canvas_result, slice_idx)
            # trigger a rerun (clears the canvas)
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()

        st.write("Stored rectangles:", st.session_state.rectangles)


def main():
    # --- 1) Load or generate your 3D image stack ---
    # Replace this with your actual data-loading logic
    image_stack = np.random.randint(0, 255, (50, 256, 256), dtype=np.uint8)

    # --- 2) Instantiate and run the annotator ---
    annotator = RectAnnotator(image_stack)
    annotator.run()


if __name__ == "__main__":
    main()
