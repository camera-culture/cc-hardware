"""Plotting utilities for sensor data visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RangeSlider, Slider

from cc_hardware.drivers.spads.spad import SPADSensor
from cc_hardware.utils.logger import get_logger


def set_matplotlib_style(*, use_scienceplots: bool = True):
    """Set the default style for matplotlib plots."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme("paper", font_scale=1.5)
    sns.set_style("ticks")

    plt.rcParams["figure.autolayout"] = True

    if use_scienceplots:
        from matplotlib import rcParams

        styles = ["science", "nature"]
        if not rcParams.get("tex.usetex", False):
            styles += ["no-latex"]

        try:
            import scienceplots  # noqa

            plt.style.use(styles)
        except ImportError:
            get_logger().warning(
                "SciencePlots not found. Using default matplotlib style."
            )


def transient_gui(
    sensor: SPADSensor,
    *,
    show: bool = True,
    save: bool = False,
    filename: str = None,
    fps: int = 10,
    fullscreen: bool = False,
    min_bin: int | None = None,
    max_bin: int | None = None,
    normalize_per_pixel: bool = False,
):
    """
    Create a GUI interface for visualizing the transient data from a SPAD sensor.

    Parameters:
        sensor: SPADSensor instance.
        show: Whether to show the GUI.
        save: Whether to save the animation to a file.
        filename: Filename to save the animation.
        fps: Base frames per second for the animation.
        fullscreen: Whether to display the GUI in fullscreen mode.
    """
    assert save or show, "Either show or save must be True."
    assert not save or not filename, "Filename must be provided if save is True."

    get_logger().info("Starting transient GUI...")

    # Set up the figure and axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)  # Make room for buttons and sliders
    if fullscreen:
        try:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        except Exception as e:
            get_logger().warning(f"Failed to set fullscreen mode: {e}")

    # Get the sensor resolution and bins
    h, w = sensor.resolution
    min_bin = min_bin or 0
    max_bin = max_bin or sensor.num_bins - 1

    # Initialize the image plot
    image_plot = ax.imshow(
        np.zeros((h, w)),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
    )
    ax.axis("off")

    # Initialize variables
    paused = [False]  # Use a list to make it mutable in nested function
    current_frame = [0]  # Mutable frame index
    num_bins = [0]  # Will be set after capturing transient
    frames = [None]  # Will hold the transient frames
    selected_bin_range = [min_bin, max_bin]  # Will be set after capturing transient

    # Add axes for the sliders
    ax_slider_bin = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_slider_fps = plt.axes([0.25, 0.1, 0.65, 0.03])

    # Initialize the bin range slider
    bin_range_slider = RangeSlider(
        ax_slider_bin,
        "Bin Range",
        valmin=0,
        valmax=num_bins[0] - 1,
        valinit=(0, num_bins[0] - 1),
        valstep=1,
    )

    # Initialize the fps slider
    fps_slider = Slider(
        ax_slider_fps,
        "FPS",
        valmin=1,
        valmax=50,
        valinit=fps,
        valstep=1,
    )

    # Function to capture a new transient
    def capture_transient(event=None):
        get_logger().info("Capturing new transient...")
        transient = sensor.accumulate(num_samples=1).T  # Note the transpose
        if not normalize_per_pixel:
            norm_transient = transient / np.max(transient)
        else:
            norm_transient = transient / np.max(transient, axis=0, keepdims=True)

        num_bins[0] = norm_transient.shape[0]
        frames[0] = [norm_transient[i, :].reshape(h, w) for i in range(num_bins[0])]
        current_frame[0] = 0  # Reset frame index

        # Update the bin range slider
        bin_range_slider.valmin = 0
        bin_range_slider.valmax = num_bins[0] - 1
        bin_range_slider.ax.set_xlim(bin_range_slider.valmin, bin_range_slider.valmax)
        bin_range_slider.set_val(selected_bin_range)

    # Initial capture
    capture_transient()

    # Update function for animation
    def update(frame_index):
        if paused[0]:
            return [image_plot]

        # Loop within the selected bin range
        if current_frame[0] > selected_bin_range[1]:
            current_frame[0] = selected_bin_range[0]

        frame_image = frames[0][current_frame[0]]
        image_plot.set_data(frame_image)

        current_frame[0] += 1
        if current_frame[0] > selected_bin_range[1]:
            current_frame[0] = selected_bin_range[0]

        return [image_plot]

    # Button callback
    def on_button_clicked(event):
        capture_transient()

    # Key press callback
    def on_key_press(event):
        if event.key == " ":
            paused[0] = not paused[0]
            get_logger().info(f"Animation {'paused' if paused[0] else 'resumed'}")
        elif event.key == "enter":
            capture_transient()

    # Slider callbacks
    def on_bin_range_change(val):
        start_bin, end_bin = map(int, bin_range_slider.val)
        selected_bin_range[0] = start_bin
        selected_bin_range[1] = end_bin
        current_frame[0] = start_bin  # Reset to start bin

    def on_fps_change(val):
        fps = fps_slider.val
        # Update the animation interval
        ani.event_source.interval = 1000 / fps

    # Connect slider callbacks
    bin_range_slider.on_changed(on_bin_range_change)
    fps_slider.on_changed(on_fps_change)

    # Create animation
    ani = FuncAnimation(
        fig,
        update,
        interval=1000 / fps,
        blit=True,
    )

    # Add the "Capture Transient" button
    ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])  # Adjusted position
    btn_capture = Button(ax_button, "Capture Transient")
    btn_capture.on_clicked(on_button_clicked)

    # Connect key press events
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    # Save the animation as mp4 if required
    if save and filename:
        if not filename.endswith(".mp4"):
            filename += ".mp4"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        get_logger().info(f"Saving animation to {filename}...")
        ani.save(filename, writer="ffmpeg", fps=fps)
        plt.close(fig)

    # Show the GUI
    if show:
        plt.show()


def plot_points(
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    x_pred: np.ndarray,
    y_pred: np.ndarray,
    check_unique: bool = False,
    fig: plt.Figure | None = None,
    filename: Path | str | None = None,
):
    if check_unique:
        # Find unique actual positions and compute the mean of the corresponding
        # predicted values
        gt_points = np.vstack([x_pos, y_pos]).T
        unique_gt, indices = np.unique(gt_points, axis=0, return_index=True)
        inv_gt, inv_ind = np.unique(gt_points, axis=0, return_inverse=True)

        x_pred_means = np.array(
            [np.median(x_pred[inv_ind == i]) for i in range(len(inv_gt))]
        )
        y_pred_means = np.array(
            [np.median(y_pred[inv_ind == i]) for i in range(len(inv_gt))]
        )

        x_pos, y_pos = unique_gt[:, 0], unique_gt[:, 1]
        x_pred, y_pred = x_pred_means, y_pred_means

        # Reorder the points in the same order as they first appear in the gt data
        x_pos = x_pos[np.argsort(indices)]
        y_pos = y_pos[np.argsort(indices)]
        x_pred = x_pred[np.argsort(indices)]
        y_pred = y_pred[np.argsort(indices)]

    # First plot: Scatter plot for X and Y positions with correspondence lines and
    # movement path
    if fig is None:
        plt.figure(figsize=(5, 5))
    plt.scatter(x_pos, y_pos, label="Actual", color="blue")
    plt.scatter(x_pred, y_pred, label="Predicted", color="orange")

    # Draw lines to show correspondence between predicted and actual points
    for x1, y1, x2, y2 in zip(x_pos, y_pos, x_pred, y_pred):
        plt.plot([x1, x2], [y1, y2], "k--", alpha=0.5)

    # Draw movement path for ground truth positions
    for i in range(1, len(x_pos)):
        plt.arrow(
            x_pos[i - 1],
            y_pos[i - 1],
            x_pos[i] - x_pos[i - 1],
            y_pos[i] - y_pos[i - 1],
            head_width=0.1,
            head_length=0.2,
            fc="blue",
            ec="blue",
            alpha=0.7,
        )

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend(loc="upper right")
    if filename is not None:
        plt.savefig(filename)
    if fig is None:
        plt.close()
