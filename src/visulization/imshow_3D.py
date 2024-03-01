import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider, TextBox


def show_3D_stack(images, number_of_slices) -> None:
    fig, ax = plt.subplots()
    plt.subplots_adjust()
    img = ax.imshow(images[number_of_slices], cmap="gray")
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])  # [left, bottom, width, height]
    slider = Slider(
        ax_slider, "Slice", 0, len(images) - 1, valinit=number_of_slices, valstep=1
    )
    ax_textbox = plt.axes([0.1, 0.05, 0.1, 0.05])
    text_box = TextBox(ax_textbox, "Slice #", initial=str(number_of_slices))

    rect = Rectangle((0, 0), 1, 1, linewidth=1, edgecolor="r", facecolor="none")
    is_drawing = False
    start_x, start_y = 0, 0
    end_x, end_y = 0, 0

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

    def on_mouse_press(event):
        global is_drawing, start_x, start_y
        if event.inaxes == ax:
            is_drawing = True
            start_x, start_y = event.xdata, event.ydata

    def on_mouse_release(event):
        global is_drawing, end_x, end_y
        if event.inaxes == ax and is_drawing:
            is_drawing = False
            end_x, end_y = event.xdata, event.ydata
            if end_x != start_x and end_y != start_y:
                rect.set_width(end_x - start_x)
                rect.set_height(end_y - start_y)
                rect.set_xy((start_x, start_y))
                fig.canvas.draw()

    def on_mouse_motion(event):
        global is_drawing, end_x, end_y
        if is_drawing and event.inaxes == ax:
            end_x, end_y = event.xdata, event.ydata
            rect.set_width(end_x - start_x)
            rect.set_height(end_y - start_y)
            rect.set_xy((start_x, start_y))
            fig.canvas.draw()

    def create_sub_image():
        global start_x, start_y, end_x, end_y
        x1, x2 = int(min(start_x, end_x)), int(max(start_x, end_x))
        y1, y2 = int(min(start_y, end_y)), int(max(start_y, end_y))
        sub_image = images[number_of_slices, y1:y2, x1:x2]
        plt.figure()
        plt.imshow(sub_image, cmap="gray")
        plt.show()

    fig.canvas.mpl_connect("button_press_event", on_mouse_press)
    fig.canvas.mpl_connect("button_release_event", on_mouse_release)
    fig.canvas.mpl_connect("motion_notify_event", on_mouse_motion)

    ax_button = plt.axes([0.1, 0.02, 0.1, 0.05])  # [left, bottom, width, height]
    button = Button(ax_button, "OK")
    button.on_clicked(create_sub_image)

    slider.on_changed(update)
    text_box.on_submit(submit)
    plt.show()
