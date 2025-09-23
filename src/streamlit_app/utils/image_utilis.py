import base64
import importlib
import io

from PIL import Image

# Patch streamlit for canvas
_st_image_mod = importlib.import_module("streamlit.elements.image")


def _image_to_url(image, *args, **kwargs):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


_st_image_mod.image_to_url = _image_to_url


def resize_if_needed(img: Image.Image, max_dim: int = 800):
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        return img.resize(new_size, Image.LANCZOS), scale
    return img, 1.0
        new_size = (int(w * scale), int(h * scale))
        return img.resize(new_size, Image.LANCZOS), scale
    return img, 1.0
