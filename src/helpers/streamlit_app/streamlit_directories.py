import os

import streamlit as st


def is_nonempty_dir(path: str) -> bool:
    return os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0


def _check_and_create_dir(path: str) -> None:
    if os.path.exists(path):
        import shutil

        shutil.rmtree(path)
        st.info(f"Overwriting existing directory: {path}")
    os.makedirs(path)
    st.success(f"Created directory: {path}")

    return None
