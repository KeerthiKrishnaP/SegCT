import os
from collections import defaultdict
from typing import Dict

from PIL import Image

from helpers.streamlit_app.streamlit_image_loader import load_images_from_dir


def load_structural_tensor_images(
    results_dir: str, slice_idx: int
) -> Dict[str, Image.Image]:
    """Load S11...S23 images for a given slice if they exist."""
    comps = ["S11", "S22", "S33", "S12", "S13", "S23"]
    images = {}
    for comp in comps:
        comp_dir = os.path.join(results_dir, comp)
        if os.path.exists(comp_dir):
            fname = f"{comp}_slice{slice_idx}.tiff"
            fpath = os.path.join(comp_dir, fname)
            if os.path.exists(fpath):
                images[comp] = Image.open(fpath)

    return images
