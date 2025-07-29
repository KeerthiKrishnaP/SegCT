# Optional patch to enable canvas background from PIL images
import base64
import importlib
import io
import os

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

from paths.image_paths import RAW_DATA
from visualization.read_image_stack import load_images_from_folder

_st_image_mod = importlib.import_module("streamlit.elements.image")


def _image_to_url(image, *args, **kwargs):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


_st_image_mod.image_to_url = _image_to_url


class CombinedROIApp:
    """
    Streamlit app for browsing a 3D image stack, drawing rectangular ROIs,
    previewing them, and batch-exporting cropped patches.
    """

    def __init__(self, img_stack: np.ndarray, canvas_key: str = "roi_canvas"):
        if img_stack.ndim not in [3, 4]:
            raise ValueError("Stack must be 3D or 4D with channels.")
        self.img_stack = img_stack
        self.depth = img_stack.shape[0]
        if "annotations" not in st.session_state:
            st.session_state.annotations = []
        self.canvas_key = canvas_key

    def run(self):
        st.title("3D Stack ROI Draw & Export")

        # Slice selection
        slice_idx = st.sidebar.slider("Slice index", 0, self.depth - 1, 0)
        slice_arr = self.img_stack[slice_idx]
        pil_img = Image.fromarray(slice_arr)

        # Drawing canvas
        st.subheader(f"Draw ROIs on Slice {slice_idx}")
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",
            stroke_width=2,
            background_image=pil_img,
            height=pil_img.height,
            width=pil_img.width,
            drawing_mode="rect",
            key=self.canvas_key,
        )
        if st.button("Add ROI"):
            self.save_rectangles(canvas_result, slice_idx)
            st.experimental_rerun()

        # List annotations
        st.subheader("Current Annotations")
        if st.session_state.annotations:
            for i, ann in enumerate(st.session_state.annotations):
                st.write(
                    f"{i}: Slice {ann['slice']}, x={ann['x']:.1f}, y={ann['y']:.1f}, w={ann['w']:.1f}, h={ann['h']:.1f}"
                )
            if st.button("Clear All"):
                st.session_state.annotations = []
        else:
            st.write("No annotations yet.")

        # Preview overlays on current slice
        disp_all = pil_img.copy()
        draw_all = ImageDraw.Draw(disp_all)
        for ann in st.session_state.annotations:
            if ann["slice"] == slice_idx:
                x0, y0 = ann["x"], ann["y"]
                x1, y1 = x0 + ann["w"], y0 + ann["h"]
                draw_all.rectangle([x0, y0, x1, y1], outline="blue", width=2)
        st.subheader("All ROIs on Current Slice")
        st.image(disp_all, use_column_width=True)

        # Preview each patch
        st.subheader("Cropped ROI Previews")
        for i, ann in enumerate(st.session_state.annotations):
            idx = ann["slice"]
            arr = self.img_stack[idx]
            patch = arr[
                int(ann["y"]) : int(ann["y"] + ann["h"]),
                int(ann["x"]) : int(ann["x"] + ann["w"]),
            ]
            img_patch = Image.fromarray(patch)
            st.image(img_patch, caption=f"{i}: Slice {idx} ROI", use_column_width=False)

        # Export
        output_dir = st.text_input("Output directory", value="output_rois")
        if st.button("Export All ROIs"):
            if not st.session_state.annotations:
                st.error("No ROIs to export.")
            else:
                count = self.crop_and_save(output_dir)
                st.success(f"Exported {count} images to '{output_dir}'")

    def save_rectangles(self, canvas_result, slice_idx: int):
        data = getattr(canvas_result, "json_data", None)
        if not data or "objects" not in data:
            return
        for obj in data["objects"]:
            if obj.get("type") == "rect":
                st.session_state.annotations.append(
                    {
                        "slice": slice_idx,
                        "x": obj["left"],
                        "y": obj["top"],
                        "w": obj["width"],
                        "h": obj["height"],
                    }
                )

    def crop_and_save(self, output_dir: str) -> int:
        os.makedirs(output_dir, exist_ok=True)
        count = 0
        for i, ann in enumerate(st.session_state.annotations):
            idx = ann["slice"]
            arr = self.img_stack[idx]
            patch_arr = arr[
                int(ann["y"]) : int(ann["y"] + ann["h"]),
                int(ann["x"]) : int(ann["x"] + ann["w"]),
            ]
            img_patch = Image.fromarray(patch_arr)
            fname = os.path.join(output_dir, f"slice_{idx}_roi_{i}.png")
            img_patch.save(fname)
            count += 1
        return count


if __name__ == "__main__":
    stack = load_images_from_folder(RAW_DATA, format=".tif")
    app = CombinedROIApp(stack)
    app.run()
