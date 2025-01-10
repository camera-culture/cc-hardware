"""SPAD dashboard based on Matplotlib for real-time visualization."""

import signal
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from cc_hardware.drivers.spads.dashboards import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils import config_wrapper, get_logger
from cc_hardware.utils.plotting import set_matplotlib_style


@config_wrapper
class MatplotlibDashboardConfig(SPADDashboardConfig):
    """
    Configuration for the Matplotlib dashboard.
    """

    instance: str = "MatplotlibDashboard"

    fullscreen: bool = False
    headless: bool = False
    save_path: Path | None = None


class MatplotlibDashboard(SPADDashboard):
    """
    Dashboard implementation using Matplotlib for visualization.
    """

    @property
    def config(self) -> MatplotlibDashboardConfig:
        return self._config

    def setup(self):
        """
        Sets up the Matplotlib plot layout and styling.
        """

        signal.signal(signal.SIGINT, signal.SIG_DFL)

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
            self.save_animation(self.ani, save_path)

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

    def save_animation(self, ani: FuncAnimation, filename: str):
        """
        Saves the animation to a file.

        Args:
            ani (FuncAnimation): The animation object to save.
            filename (str): The filename to save the output.
        """
        import matplotlib.pyplot as plt

        get_logger().info("Saving animation...")
        if not filename.endswith(".mp4"):
            filename += ".mp4"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        get_logger().info(f"Saving animation to {filename}...")
        ani.save(filename, writer="ffmpeg", dpi=400, fps=10)
        get_logger().info("Saved animation.")
        plt.close(self.fig)
