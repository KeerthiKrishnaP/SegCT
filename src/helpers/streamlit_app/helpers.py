import os

import streamlit as st


def get_input_directory(fetch_option: str, operation: str) -> None:
    if fetch_option == "Auto-fetch":
        # Example: automatically fetch from default location
        default_dir = f"./data/{operation.lower()}"
        st.info(f"Auto-fetching inputs from: `{default_dir}`")

        if os.path.exists(default_dir):
            files = os.listdir(default_dir)
            st.success(f"Found {len(files)} files.")
        else:
            st.warning("Default directory not found.")

    elif fetch_option == "Enter directory path":
        if user_path := st.text_input("Enter the full directory path:"):
            if os.path.exists(user_path):
                files = os.listdir(user_path)
                st.success(f"Loaded {len(files)} files from `{user_path}`.")
            else:
                st.error("Path not found. Please check and try again.")
