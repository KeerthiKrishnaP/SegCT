import streamlit as st

import helpers.streamlit_app.page1_crop as page1_crop
import helpers.streamlit_app.page2_annotate_ROI as page2_annotator
import helpers.streamlit_app.page3_computations as page3_computations
import helpers.streamlit_app.page4_features as page4_features
import helpers.streamlit_app.page6_viewer as page6_viewer


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
        page1_crop.app()

    elif page == "Page 2: Annotate ROIs":
        page2_annotator.app()

    elif page == "Page 3: Computations":
        page3_computations.app()

    elif page == "Page 4: Features":
        page4_features.app()

    elif page == "Page 5: Segmentation":
        st.info("Viewer page (coming soon).")

    elif page == "Page 6: Viewer":
        page6_viewer.app()


if __name__ == "__main__":
    main()
