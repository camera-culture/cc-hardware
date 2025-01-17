"""SPAD dashboard based on Matplotlib for real-time visualization."""

from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RangeSlider, Slider

from cc_hardware.tools.dashboard.spad_dashboard import (
    SPADDashboard,
    SPADDashboardConfig,
)
from cc_hardware.utils import config_wrapper, get_logger
from cc_hardware.utils.matplotlib import set_matplotlib_style


def save_animation(ani: FuncAnimation, filename: str):
    """
    Saves the animation to a file.

    Args:
        ani (FuncAnimation): The animation object to save.
        filename (str): The filename to save the output.
    """

    get_logger().info("Saving animation...")
    if not filename.endswith(".mp4"):
        filename += ".mp4"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    get_logger().info(f"Saving animation to {filename}...")
    ani.save(filename, writer="ffmpeg", dpi=400, fps=10)
    get_logger().info("Saved animation.")


@config_wrapper
class MatplotlibDashboardConfig(SPADDashboardConfig):
    """
    Configuration for the Matplotlib dashboard.
    """

    fullscreen: bool = False
    headless: bool = False
    save_path: Path | None = None


class MatplotlibDashboard(SPADDashboard[MatplotlibDashboardConfig]):
    """
    Dashboard implementation using Matplotlib for visualization.
    """

    def setup(self):
        """
        Sets up the Matplotlib plot layout and styling.
        """

        set_matplotlib_style()

        rows = int(np.ceil(np.sqrt(self.num_channels)))
        cols = int(np.ceil(self.num_channels / rows))

        self.fig = plt.figure(figsize=(6, 6 * rows / cols))
        self.gs = plt.GridSpec(rows, cols, figure=self.fig)

        self.bins = np.arange(self.min_bin, self.max_bin)
        colors = plt.cm.viridis(np.linspace(0, 1, self.num_channels))

        self.axes = []
        self.containers = []
        min_bin, max_bin = self.min_bin, self.max_bin
        for idx, _ in enumerate(self.channel_mask):
            row, col = divmod(idx, cols)
            ax = self.fig.add_subplot(self.gs[row, col])
            self.axes.append(ax)
            container = ax.hist(
                self.bins,
                bins=self.bins.size,
                weights=np.zeros(self.bins.size),
                edgecolor="black",
                color=colors[idx],
            )[2]
            self.containers.append(container)
            ax.set_xlim(min_bin, max_bin)
            ax.set_xlabel("Bin")
            ax.set_ylabel("Photon Counts")

        plt.tight_layout()

        if self.config.fullscreen:
            try:
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle()
            except Exception as e:
                get_logger().warning(f"Failed to set fullscreen mode: {e}")

        self.save_path = self.config.save_path
        self.headless = self.config.headless

    def run(self):
        """
        Executes the Matplotlib dashboard with real-time updates.
        """
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            frames=range(self.config.num_frames),
            interval=1,
            repeat=False,
            blit=True,
        )

        if self.save_path:
            # Prevent saving on each frame
            save_path = self.save_path
            self.save_path = None
            save_animation(self.ani, save_path)

        if not self.headless:
            plt.show()

    def update(self, frame: int, *, histograms: np.ndarray | None = None):
        """
        Updates the histogram data for each frame.

        Args:
            frame (int): Current frame number.

        Keyword Args:
            histograms (np.ndarray): The histogram data to update. If not provided, the
                sensor will be used to accumulate the histogram data.
        """
        import matplotlib.pyplot as plt

        if not plt.fignum_exists(self.fig.number):
            get_logger().info("Closing GUI...")
            return

        # Check if the user has updated the number of bins
        if self._sensor.num_bins != self.bins.size:
            # Update x-axis
            self.bins = np.arange(self._sensor.num_bins + 1)  # Adjust bin range
            for container in self.containers:
                for rect, x in zip(container, self.bins[:-1]):
                    rect.set_x(x)  # Update x-coordinates of histogram bars
            for ax in self.axes:
                ax.set_xlim(0, self._sensor.num_bins)  # Update x-axis limits
                ax.set_xticks(
                    np.linspace(0, self._sensor.num_bins, 5)
                )  # Update x-ticks

        if histograms is None:
            histograms = self._sensor.accumulate()
        self.adjust_ylim(histograms)

        for idx, channel in enumerate(self.channel_mask):
            container = self.containers[idx]
            histogram = histograms[channel, self.min_bin : self.max_bin]
            for rect, h in zip(container, histogram):
                rect.set_height(h)

        # Call user callback if provided
        if self.config.user_callback is not None:
            self.config.user_callback(self)

        if self.save_path:
            self.fig.savefig(self.save_path / "frame_{frame}.png")
        elif not self.headless:
            plt.pause(1e-9)

        return list(chain(*self.containers))

    def adjust_ylim(self, histograms: np.ndarray):
        """
        Adjusts the Y-axis limits based on the histogram data.

        Args:
            histograms (np.ndarray): Histogram data for all channels.
        """
        if self.config.ylim is not None:
            for ax in self.axes:
                ax.set_ylim(0, self.config.ylim)
        elif self.config.autoscale:
            for ax, idx in zip(self.axes, self.channel_mask):
                histogram = histograms[idx, self.min_bin : self.max_bin]
                ax.set_ylim(0, histogram.max())
        else:
            max_count = np.max(
                [
                    histograms[channel, self.min_bin : self.max_bin].max()
                    for channel in self.channel_mask
                ]
            )
            for ax in self.axes:
                ax.set_ylim(0, max_count)

    @property
    def is_okay(self) -> bool:
        return hasattr(self, "fig") and plt.fignum_exists(self.fig.number)

    def close(self) -> None:
        if self.is_okay:
            plt.close(self.fig)


@config_wrapper
class MatplotlibTransientViewerConfig(SPADDashboardConfig):
    fullscreen: bool = False
    headless: bool = False
    save_path: Path | None = None

    normalize_per_pixel: bool = False


class MatplotlibTransientViewer(SPADDashboard[MatplotlibTransientViewerConfig]):
    def setup(self):
        set_matplotlib_style()

        # Setup the figure and axes
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)  # Make room for buttons and sliders

        # Initialize the image plot
        h, w = self._sensor.resolution
        self.image_plot = self.ax.imshow(
            np.zeros((h, w)),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="auto",
        )
        self.ax.axis("off")

        # Add axes for the sliders
        self.ax_slider_bin = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.bin_range_slider = RangeSlider(
            self.ax_slider_bin,
            "Bin Range",
            valmin=0,
            valmax=self.sensor.num_bins - 1,
            valinit=(0, self.sensor.num_bins - 1),
            valstep=1,
        )

        def on_bin_range_change(val):
            self.min_bin, self.max_bin = map(int, self.bin_range_slider.val)

        self.bin_range_slider.on_changed(on_bin_range_change)

        self.ax_slider_fps = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.fps_slider = Slider(
            self.ax_slider_fps,
            "FPS",
            valmin=1,
            valmax=50,
            valinit=10,
            valstep=1,
        )

        def on_fps_change(val):
            self.ani.event_source.interval = 1000 / self.fps_slider.val

        self.fps_slider.on_changed(on_fps_change)

        # Add the "Capture Transient" button
        self.ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])  # Adjusted position
        self.btn_capture = Button(self.ax_button, "Capture Transient")

        def on_button_clicked(event):
            self.update(-1, histograms=self.sensor.accumulate())

        self.btn_capture.on_clicked(on_button_clicked)

        if self.config.fullscreen:
            try:
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle()
            except Exception as e:
                get_logger().warning(f"Failed to set fullscreen mode: {e}")

    def run(self):
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            interval=1000 / self.fps_slider.val,
            blit=True,
        )

        if self.save_path:
            # Prevent saving on each frame
            save_path = self.save_path
            self.save_path = None
            save_animation(self.ani, save_path)

        if not self.config.headless:
            plt.show()

    def update(self, frame_index: int, *, histograms: np.ndarray | None = None):
        if not plt.fignum_exists(self.fig.number):
            get_logger().info("Closing GUI...")
            return

        if histograms is None:
            histograms = self._sensor.accumulate()
        transient = histograms.T

        if not self.config.normalize_per_pixel:
            norm_transient = transient / np.max(transient)
        else:
            norm_transient = transient / np.max(transient, axis=0, keepdims=True)

        self.image_plot.set_data(norm_transient[frame_index % norm_transient.shape[0]])

        # Update the bin range slider
        self.bin_range_slider.valmin = 0
        self.bin_range_slider.valmax = self.sensor.num_bins - 1
        self.bin_range_slider.ax.set_xlim(
            self.bin_range_slider.valmin, self.bin_range_slider.valmax
        )
        self.bin_range_slider.set_val((0, self.sensor.num_bins - 1))
