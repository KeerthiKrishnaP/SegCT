import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RectangleSelector, Slider, TextBox
from PIL import Image


class ImageViewer:
    def __init__(self, images):
        self.images = images
        self.starting_slice = 0  # Starting slice index
        self.final_slice = len(images) - 1  # Final slice index
        self.rectangle_coords = None
        self.figure, self.axis = plt.subplots()
        self.axis.set_title("Image Viewer")
        self.image_to_show = self.axis.imshow(images[self.starting_slice], cmap="gray")
        self.colorbar = self.figure.colorbar(
            self.image_to_show, ax=self.axis, orientation="horizontal"
        )
        self.colorbar.set_label("Intensity")
        self.slider_bar = plt.axes((0.1, 0.1, 0.8, 0.03))
        self.slider = Slider(
            self.slider_bar,
            "Range",
            0,
            len(images) - 1,
            valinit=self.starting_slice,
            valstep=1,
        )
        self.image_number = plt.axes((0.11, 0.01, 0.12, 0.05))
        self.image_text_box = TextBox(
            self.image_number, "Slice #", initial=str(self.starting_slice)
        )
        self.rectangle_selector = RectangleSelector(
            self.axis,
            self.onselect,
            useblit=True,
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        self.slider.on_changed(self.update)
        self.image_text_box.on_submit(self.submit)
        self.starting_text_box = TextBox(
            plt.axes((0.2, 0.8, 0.1, 0.04)), "Start:", initial=str(self.starting_slice)
        )
        self.final_text_box = TextBox(
            plt.axes((0.2, 0.75, 0.1, 0.04)), "End:", initial=str(self.final_slice)
        )
        self.name_of_sub_image = TextBox(
            plt.axes((0.2, 0.7, 0.2, 0.04)),
            "Dataset name:",
            initial="None",
        )
        self.save_button = Button(plt.axes((0.4, 0.8, 0.15, 0.04)), "Save Dataset")
        self.save_button.on_clicked(self.save_sub_images)

    def onselect(self, eclick, erelease):
        self.rectangle_coords = (
            eclick.xdata,
            eclick.ydata,
            erelease.xdata,
            erelease.ydata,
        )

    def draw_rectangle(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        rect = Rectangle(
            (min(x1, x2), min(y1, y2)),
            abs(x2 - x1),
            abs(y2 - y1),
            edgecolor="r",
            alpha=0.5,
        )
        self.axis.add_patch(rect)
        self.figure.canvas.draw_idle()

    def update(self, value):
        self.starting_slice = int(self.slider.val)
        self.final_slice = min(len(self.images) - 1, self.starting_slice + 1)
        self.image_to_show.set_data(self.images[self.starting_slice])
        self.figure.canvas.draw_idle()

    def submit(self, slice_number_in_text):
        try:
            if 0 <= int(slice_number_in_text) < len(self.images):
                self.slider.set_val(int(slice_number_in_text))
                self.image_to_show.set_data(self.images[int(slice_number_in_text)])
                self.figure.canvas.draw_idle()
            else:
                print("Slice number out of range.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    def save_sub_images(self, event):
        path = f"src/datasets/training_sets/{self.name_of_sub_image.text}"
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        if self.rectangle_coords is not None:
            start_x, end_x = int(round(self.rectangle_coords[0])), int(
                round(self.rectangle_coords[2])
            )
            start_y, end_y = int(round(self.rectangle_coords[1])), int(
                round(self.rectangle_coords[3])
            )
            print(int(self.starting_text_box.text))
            print(self.rectangle_coords)
            for i in range(
                int(self.starting_text_box.text),
                int(self.final_text_box.text),
            ):
                sub_image = Image.fromarray(
                    np.array(self.images[i])[start_y:end_y, start_x:end_x]
                )
                sub_image.save(f"{path}/sub_image_{i}.png")
                print(f"Sub-image {i} saved.")
        else:
            print("Please select a rectangle before saving sub-images.")
