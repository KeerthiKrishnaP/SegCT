import streamlit as st

import helpers.streamlit_app.page2_annotate_ROI as roi_annotator
import helpers.streamlit_app.page3_computations as computations
import helpers.streamlit_app.page4_features as features
import helpers.streamlit_app.page6_viewer as viewer
import src.helpers.streamlit_app.page1_load_crop_main_image as load_crop


def main() -> None:
    st.title("3D Image Stack Workflow")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Page 1: Crop Images",
            "Page 2: Annotate ROIs",
            "Page 3: Computations",
            "Page 4: Features",
            "Page 5: Segmentation",
            "Page 6: Viewer",
        ],
    )

    if page == "Page 1: Crop Images":
        load_crop.app()

    elif page == "Page 2: Annotate ROIs":
        roi_annotator.app()

    elif page == "Page 3: Computations":
        computations.app()

    elif page == "Page 4: Features":
        features.app()

    elif page == "Page 5: Segmentation":
        st.info("Viewer page (coming soon).")

    elif page == "Page 6: Viewer":
        viewer.app()


if __name__ == "__main__":
    main()
