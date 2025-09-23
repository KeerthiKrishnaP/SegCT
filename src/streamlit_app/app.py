import streamlit as st

from .pages.paceholders import segmentation_page, viewer_page
from .pages.page1_crop import page1_crop_images
from .pages.page2_annotate import run_page2
from .pages.page3_compute import computations_page


def main():
    st.title("3D Image Stack Workflow")

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

    if page == "Page 1: Crop Images":
        page1_crop_images()
    elif page == "Page 2: Annotate ROIs":
        run_page2()
    elif page == "Page 3: Computations":
        computations_page()
    elif page == "Page 4: Segmentation & Stats":
        segmentation_page()
    elif page == "Page 5: Viewer":
        viewer_page()


if __name__ == "__main__":
    main()
    main()
