import base64
import importlib
import io
import os
import shutil

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.computations.comput_features import parallel_structural_tensor

# ───────────────────────────────────────────
# Patch to use image_to_url for canvas background
_st_image_mod = importlib.import_module("streamlit.elements.image")


def _image_to_url(image, *args, **kwargs):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


_st_image_mod.image_to_url = _image_to_url
# ───────────────────────────────────────────


def resize_if_needed(img: Image.Image, max_dim: int = 800):
    """Resize image to fit within max_dim, return resized + scale factors."""
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        return img.resize(new_size, Image.LANCZOS), scale
    return img, 1.0


class Cropper:
    def __init__(self, image_stack: np.ndarray, save_dir: str, prefix="page1_cropper"):
        self.image_stack = image_stack
        self.save_dir = save_dir
        self.prefix = prefix
        self.crop_dir = os.path.join(save_dir, "crops", "data_stack")
        os.makedirs(self.crop_dir, exist_ok=True)

    def select_slice(self):
        return st.slider(
            "Select slice for cropping reference",
            min_value=0,
            max_value=self.image_stack.shape[0] - 1,
            value=0,
            key=f"{self.prefix}_slice_slider",
        )

    def draw_crop_rect(self, image: np.ndarray):
        img_pil = Image.fromarray(image)
        resized_img, scale = resize_if_needed(img_pil)
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",
            stroke_width=2,
            background_image=resized_img,
            height=resized_img.size[1],
            width=resized_img.size[0],
            drawing_mode="rect",
            key=f"{self.prefix}_canvas",
        )
        return canvas_result, scale, img_pil.size

    def crop_all(self, rect_coords, start_idx, end_idx):
        x, y, w, h = rect_coords
        for i in range(start_idx, end_idx + 1):
            img = Image.fromarray(self.image_stack[i])
            cropped = img.crop((x, y, x + w, y + h))
            filename = f"cropped_slice{i}.tiff"
            save_path = os.path.join(self.crop_dir, filename)
            cropped.save(save_path)
        st.success(f"Saved cropped images {start_idx} → {end_idx} to {self.crop_dir}")

    def run(self):
        slice_idx = self.select_slice()
        img = self.image_stack[slice_idx]
        canvas_result, scale, orig_size = self.draw_crop_rect(img)

        # Pick slice range for cropping
        st.write("### Slice Range for Cropping")
        start_idx = st.number_input(
            "Start slice index",
            min_value=0,
            max_value=self.image_stack.shape[0] - 1,
            value=0,
            key=f"{self.prefix}_start_idx",
        )
        end_idx = st.number_input(
            "End slice index",
            min_value=0,
            max_value=self.image_stack.shape[0] - 1,
            value=self.image_stack.shape[0] - 1,
            key=f"{self.prefix}_end_idx",
        )

        if st.button("Crop All Images", key=f"{self.prefix}_crop_button"):
            data = canvas_result.json_data
            if data and "objects" in data and len(data["objects"]) > 0:
                rect = data["objects"][0]  # take first rect
                # Map rect coords back to original resolution
                x = int(rect["left"] / scale)
                y = int(rect["top"] / scale)
                w = int(rect["width"] / scale)
                h = int(rect["height"] / scale)
                self.crop_all((x, y, w, h), int(start_idx), int(end_idx))
            else:
                st.warning("Please draw a rectangle before cropping.")


def page1_crop_images():
    st.header("Load & Crop Images")

    # --- Choose mode: create or load ---
    mode = st.radio(
        "Choose option:",
        ["Create New Working Directory", "Load Existing Working Directory"],
        key="page1_mode_radio",
    )

    working_dir = None

    if mode == "Create New Working Directory":
        base_dir = st.text_input(
            "Enter base path:",
            os.path.expanduser("~"),
            key="page1_base_dir_input",
        )

        new_dir_name = st.text_input(
            "Enter a name for the working directory:",
            "working_output",
            key="page1_new_dir_name_input",
        )
        working_dir = os.path.join(base_dir, new_dir_name)
        if os.path.exists(working_dir):
            st.warning("Directory already exists. Its being overwritten.")
        if st.button("Create Working Directory", key="page1_create_dir_button"):
            if os.path.exists(working_dir) and os.path.isdir(working_dir):
                shutil.rmtree(working_dir)
            os.makedirs(working_dir)
            st.session_state["working_dir"] = working_dir
            st.success(
                f"Created working directory: {os.path.abspath(working_dir)}", icon="📁"
            )

    elif mode == "Load Existing Working Directory":
        existing_dir = st.text_input(
            "Enter path to existing working directory:",
            key="page1_existing_dir_input",
        )

        if st.button("Load Working Directory", key="page1_load_dir_button"):
            if os.path.exists(existing_dir) and os.path.isdir(existing_dir):
                working_dir = existing_dir
                st.session_state["working_dir"] = working_dir
                st.success(
                    f"Loaded working directory: {os.path.abspath(working_dir)}",
                    icon="📂",
                )
            else:
                st.error("❌ The path does not exist or is not a directory.")

    if "working_dir" in st.session_state:
        uploaded_files = st.file_uploader(
            "Upload image stack",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
            key="page1_file_uploader",
        )

        if uploaded_files:
            image_list = [np.array(Image.open(f).convert("L")) for f in uploaded_files]
            image_stack = np.stack(image_list, axis=0)
            cropper = Cropper(
                image_stack, st.session_state["working_dir"], prefix="page1_cropper"
            )
            cropper.run()
    else:
        st.info("Please create or load a working directory first.", icon="ℹ️")


class RectAnnotator:
    def __init__(
        self, image_stack: np.ndarray, save_dir: str, prefix="page2_annotator"
    ):
        self.image_stack = image_stack
        self.save_dir = save_dir
        self.prefix = prefix

    def select_slice(self) -> int:
        return st.slider(
            "Select slice (for drawing ROI boundary)",
            min_value=0,
            max_value=self.image_stack.shape[0] - 1,
            value=0,
            key=f"{self.prefix}_slice_slider",
        )

    def draw_canvas(self, image: np.ndarray):
        img_pil = Image.fromarray(image)
        resized_img, scale = resize_if_needed(img_pil)
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 255, 0.2)",
            stroke_width=2,
            background_image=resized_img,
            height=resized_img.size[1],
            width=resized_img.size[0],
            drawing_mode="rect",
            key=f"{self.prefix}_canvas",
        )
        return canvas_result, scale, img_pil.size

    def save_dataset(self, rect_coords, start_idx, end_idx, dataset_name, dataset_num):
        # Create dataset directory
        dataset_dir = os.path.join(self.save_dir, dataset_name, str(dataset_num))
        os.makedirs(dataset_dir, exist_ok=True)

        x, y, w, h = rect_coords
        for i in range(start_idx, end_idx + 1):
            img = Image.fromarray(self.image_stack[i])
            cropped = img.crop((x, y, x + w, y + h))
            filename = f"roi_slice{i}.png"
            cropped.save(os.path.join(dataset_dir, filename))

        # Save metadata file
        info_path = os.path.join(dataset_dir, "roi_info.txt")
        with open(info_path, "w") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Number: {dataset_num}\n")
            f.write(f"Rectangle: x={x}, y={y}, w={w}, h={h}\n")
            f.write(f"Start slice: {start_idx}\n")
            f.write(f"End slice: {end_idx}\n")

        st.success(
            f"Saved slices {start_idx} → {end_idx} and roi_info.txt to {dataset_dir}"
        )

    def run(self):
        # Step 1: Pick a slice to draw ROI
        slice_idx = self.select_slice()
        img = self.image_stack[slice_idx]
        canvas_result, scale, _ = self.draw_canvas(img)

        # Step 2: User inputs
        st.write("### ROI Dataset Parameters")
        start_idx = st.number_input(
            "Start slice index",
            min_value=0,
            max_value=self.image_stack.shape[0] - 1,
            value=0,
            key=f"{self.prefix}_start_idx",
        )
        end_idx = st.number_input(
            "End slice index",
            min_value=0,
            max_value=self.image_stack.shape[0] - 1,
            value=self.image_stack.shape[0] - 1,
            key=f"{self.prefix}_end_idx",
        )
        dataset_name = st.text_input(
            "Dataset name (e.g., warp):", "warp", key=f"{self.prefix}_dataset_name"
        )
        dataset_num = st.number_input(
            "Dataset number:", min_value=1, value=1, key=f"{self.prefix}_dataset_num"
        )

        # Step 3: Save
        if st.button("Save Dataset", key=f"{self.prefix}_save_button"):
            data = canvas_result.json_data
            if data and "objects" in data and len(data["objects"]) > 0:
                rect = data["objects"][0]  # first rect
                # Convert back to original resolution
                x = int(rect["left"] / scale)
                y = int(rect["top"] / scale)
                w = int(rect["width"] / scale)
                h = int(rect["height"] / scale)
                self.save_dataset(
                    (x, y, w, h),
                    int(start_idx),
                    int(end_idx),
                    dataset_name,
                    dataset_num,
                )
            else:
                st.warning("Please draw a rectangle before saving.")


def load_cropped_images(save_dir):
    crop_dir = os.path.join(save_dir, "crops", "data_stack")
    if not os.path.exists(crop_dir):
        return None
    files = sorted([f for f in os.listdir(crop_dir) if f.endswith(".tiff")])
    if not files:
        return None
    image_list = [
        np.array(Image.open(os.path.join(crop_dir, f)).convert("L")) for f in files
    ]
    return np.stack(image_list, axis=0)


def load_images(dir):
    if not os.path.exists(dir):
        return None
    files = sorted([f for f in os.listdir(dir) if f.endswith(".tiff")])
    if not files:
        return None
    image_list = [
        np.array(Image.open(os.path.join(dir, f)).convert("L")) for f in files
    ]
    return np.stack(image_list, axis=0)


def is_nonempty_dir(path):
    return os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0


def load_structural_tensor_images(results_dir, slice_idx):
    """Load S11...S23 images for a given slice if they exist."""
    comps = ["S11", "S22", "S33", "S12", "S13", "S23"]
    images = {}
    for comp in comps:
        comp_dir = os.path.join(results_dir, comp)
        if os.path.exists(comp_dir):
            fname = f"{comp}_slice{slice_idx}.png"
            fpath = os.path.join(comp_dir, fname)
            if os.path.exists(fpath):
                images[comp] = Image.open(fpath)
    return images


def check_and_create_dir(path) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
        st.info(f"Overwriting existing directory: {path}")
        os.makedirs(path)


def computations_page():
    st.header("Segmentation and Computations")

    if "working_dir" not in st.session_state:
        st.warning("Please create or load a working directory in Page 1 first.")
        return

    base_dir = st.session_state["working_dir"]

    # Browse dataset
    dataset_path = st.text_input(
        "Enter dataset path (relative to working directory):",
        key="page3_dataset_path_input",
    )
    data_path = os.path.join(base_dir, dataset_path)
    if not os.path.exists(data_path):
        st.warning("Dataset path not found. Example: warp/1 or crops/")
        return

    # Results check
    results_struct_dir = os.path.join(base_dir, "results_structural_tensor")
    results_average_gray_value_dir = os.path.join(
        base_dir, "results_average_gray_value"
    )
    results_azmital_angle_dir = os.path.join(base_dir, "results_azimuthal_angle")

    results_struct_done = is_nonempty_dir(results_struct_dir)
    results_average_gray_value_done = is_nonempty_dir(results_average_gray_value_dir)
    results_azmital_angle_done = is_nonempty_dir(results_azmital_angle_dir)

    # Choose computation
    comp_type = st.selectbox(
        "Choose computation to run:",
        [
            f"Structural Tensor {'✅' if results_struct_done else ''}",
            f"Average Gray Value {'✅' if results_average_gray_value_done else ''}",
            f"Azimuthal Angle {'✅' if results_azmital_angle_done else ''}",
        ],
        key="page3_comp_type",
    )

    # Common params
    st.write("### Common Parameters")
    window_radius = st.number_input(
        "Window radius:", min_value=1, value=5, key="page3_window_radius"
    )
    window_size = st.number_input(
        "Window size:", min_value=1, value=15, key="page3_window_size"
    )
    parallel = st.checkbox("Run in parallel?", value=False, key="page3_parallel_flag")
    st.info(f"Dataset loaded from {dataset_path} ...")
    image = load_images(data_path)

    if image is None:
        st.error("No cropped images found in the dataset path.")
        return
    # Structural Tensor UI
    match comp_type:
        case "Structural Tensor":
            st.subheader("Structural Tensor Computation")
            if st.button("Run Computation", key="page3_run_button"):
                # Run computation
                st.info(f"Running Structural Tensor computation on {dataset_path} ...")
                results = parallel_structural_tensor(image, window_size, parallel)
                # Save results
                for components, components_image_array in results.items():
                    components_dir = os.path.join(results_struct_dir, components)
                    os.makedirs(components, exist_ok=True)
                    for i in range(components_image_array.shape[0]):  # save each slice
                        img = Image.fromarray(
                            (components_image_array[i] * 255).astype(np.uint8)
                        )
                        fname = f"{components}_slice{i}.png"
                        img.save(os.path.join(components_dir, fname))
                st.success(
                    f"✅ Structural Tensor results saved in {results_struct_dir}"
                )

            # If results exist → visualization
            if results_struct_done:
                st.markdown("### Structural Tensor Results")
                sample_comp = os.path.join(results_struct_dir, "S11")
                if os.path.exists(sample_comp):
                    if slice_files := [
                        f for f in os.listdir(sample_comp) if f.endswith(".png")
                    ]:
                        max_idx = len(slice_files) - 1
                        slice_idx = st.slider(
                            "Select slice", 0, max_idx, 0, key="page3_slice_slider"
                        )
                        if images := load_structural_tensor_images(
                            results_struct_dir, slice_idx
                        ):
                            cols = st.columns(len(images))
                            for i, (comp, img) in enumerate(images.items()):
                                with cols[i]:
                                    st.image(
                                        img,
                                        caption=f"{comp} - slice {slice_idx}",
                                        use_container_width=True,
                                    )
                        else:
                            st.info("No images found for selected slice.")
        case "Average Gray Value":
            st.info("Average Gray Value computation")
            if st.button("Run Computation", key="page3_run_button"):
                st.info(f"Running Average Gray Value computation on {dataset_path} ...")
                # Run computation
                # results = average_gray_value(image, window_size)
                # Save results
                # comp_dir = os.path.join(results_average_gray_value_dir, "AverageGrayValue")
                # os.makedirs(comp_dir, exist_ok=True)
                # for i in range(results.shape[0]):  # save each slice
                #     img = Image.fromarray((results[i] * 255).astype(np.uint8))
                #     fname = f"AverageGrayValue_slice{i}.png"
                #     img.save(os.path.join(comp_dir, fname))

                st.success(
                    f"✅ Average Gray Value results saved in {results_average_gray_value_dir}"
                )

        case "Azimuthal Angle":
            st.info("Azimuthal Angle computation coming soon.")


# ───────────────────────────────────────────
# Main App
# ───────────────────────────────────────────
def main():
    st.title("3D Image Stack Workflow")

    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        [
            "Page 1: Crop Images",
            "Page 2: Annotate ROIs",
            "Page 3: Computations",
            "Page 4: Segmentation & Stats",
            "Page 5: Viewer",
        ],
        key="sidebar_nav_radio",
    )

    # ───────────────────────────────────────
    # Page 1: Crop images
    if page == "Page 1: Crop Images":
        st.header("Load & Crop Images")
        page1_crop_images()
    # ───────────────────────────────────────
    elif page == "Page 2: Annotate ROIs":
        st.header("Annotate Regions of Interest")

        if "working_dir" not in st.session_state:
            st.warning("Please create a working directory in Page 1 first.")
            return

        cropped_stack = load_cropped_images(st.session_state["working_dir"])
        if cropped_stack is None:
            st.warning(
                "No cropped images found. Please perform cropping in Page 1 first."
            )
        else:
            annotator = RectAnnotator(
                cropped_stack, st.session_state["working_dir"], prefix="page2_annotator"
            )
            annotator.run()

    elif page == "Page 3: Computations":
        st.header(
            "Compute parameters that will be subsequently be used for segmentation"
        )
        computations_page()

    elif page == "Page 4: Segmentation & Stats":
        st.header("Segmentation and Analysis")
        st.info("This page will run segmentation and show stats/plots (coming soon).")

    elif page == "Page 5: Viewer":
        st.header("View 3D Stack and plots histograms")
        st.info("(coming soon).")


if __name__ == "__main__":
    main()
