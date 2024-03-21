import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider, TextBox


def show_3D_stack(images, number_of_slices) -> None:
    fig, ax = plt.subplots()
    plt.subplots_adjust()
    img = ax.imshow(images[number_of_slices], cmap="gray")
    ax_slider = plt.axes((0.1, 0.1, 0.8, 0.03))  # [left, bottom, width, height]
    slider = Slider(
        ax_slider, "Slice", 0, len(images) - 1, valinit=number_of_slices, valstep=1
    )
    ax_textbox = plt.axes((0.1, 0.05, 0.1, 0.05))
    text_box = TextBox(ax_textbox, "Slice #", initial=str(number_of_slices))

    def update(val):
        current_slice = int(slider.val)
        img.set_data(images[current_slice])
        fig.canvas.draw_idle()

    def submit(text):
        try:
            new_slice = int(text)
            if 0 <= new_slice < len(images):
                current_slice = new_slice
                slider.set_val(current_slice)
                img.set_data(images[current_slice])
                fig.canvas.draw_idle()
            else:
                print("Slice number out of range.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    slider.on_changed(update)
    text_box.on_submit(submit)
    plt.show()
