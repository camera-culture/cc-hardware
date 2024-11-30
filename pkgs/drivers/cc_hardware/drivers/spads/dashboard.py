"""Dashboard for SPAD sensors."""

import signal
import threading
import time
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path

import numpy as np

from cc_hardware.drivers.spads.spad import SPADSensor
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.registry import Registry, register

# ================


class SPADDashboard(ABC, Registry):
    """
    Abstract base class for SPAD sensor dashboards.

    Parameters:
        sensor (SPADSensor): The SPAD sensor instance.
        num_frames (int): Number of frames to process. Default is 1,000,000.
        show (bool): Whether to display the dashboard. Default is True.
        save (bool): Whether to save the output. Default is False.
        filename (str, optional): Filename to save the output if `save` is True.
        min_bin (int, optional): Minimum bin value for histogram.
        max_bin (int, optional): Maximum bin value for histogram.
        autoscale (bool): Whether to autoscale the histogram. Default is True.
        ylim (float, optional): Y-axis limit for the histogram.
        channel_mask (list[int], optional): List of channels to display.
        fullscreen (bool): Whether to display in fullscreen mode. Default is False.
    """

    def __init__(
        self,
        sensor: SPADSensor,
        num_frames: int = 1_000_000,
        show: bool = True,
        save: bool = False,
        filename: str | None = None,
        min_bin: int | None = None,
        max_bin: int | None = None,
        autoscale: bool = True,
        ylim: float | None = None,
        channel_mask: list[int] | None = None,
        fullscreen: bool = False,
    ):
        self.sensor = sensor
        self.num_frames = num_frames
        self.show = show
        self.save = save
        self.filename = filename
        self.min_bin = min_bin or 0
        self.max_bin = max_bin or sensor.num_bins - 1
        self.autoscale = autoscale
        self.ylim = ylim
        self.channel_mask = channel_mask
        self.fullscreen = fullscreen

        self.validate_parameters()
        self.setup_sensor()
        get_logger().info("Starting histogram GUI...")

    def validate_parameters(self):
        """
        Validates the initialization parameters to ensure correct usage.
        """
        assert self.save or self.show, "Either show or save must be True."
        if self.save and not self.filename:
            raise ValueError("Filename must be provided if save is True.")

    def setup_sensor(self):
        """
        Configures the sensor settings and channel mask.
        """
        h, w = self.sensor.resolution
        total_channels = h * w
        if self.channel_mask is None:
            self.channel_mask = np.arange(total_channels)
        self.channel_mask = np.array(self.channel_mask)
        self.num_channels = len(self.channel_mask)

    @abstractmethod
    def run(self):
        """
        Abstract method to run the dashboard.
        """
        pass


# ================


@register
class MatplotlibDashboard(SPADDashboard):
    """
    Dashboard implementation using Matplotlib for visualization.
    """

    def run(self):
        """
        Executes the Matplotlib dashboard with real-time updates.
        """
        global plt, Slider
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import Slider

        from cc_hardware.utils.plotting import set_matplotlib_style

        set_matplotlib_style()

        self.setup_plot()
        ani = FuncAnimation(
            self.fig,
            self.update,
            frames=range(self.num_frames),
            interval=1,
            repeat=False,
            blit=True,
        )
        if self.show:
            plt.show()
        if self.save and self.filename:
            self.save_animation(ani)

    def setup_plot(self):
        """
        Sets up the Matplotlib plot layout and styling.
        """
        rows = int(np.ceil(np.sqrt(self.num_channels)))
        cols = int(np.ceil(self.num_channels / rows))

        self.fig = plt.figure(figsize=(6, 6 * rows / cols))
        self.gs = plt.GridSpec(rows, cols, figure=self.fig)

        if self.fullscreen:
            try:
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle()
            except Exception as e:
                get_logger().warning(f"Failed to set fullscreen mode: {e}")

        self.bins = np.arange(self.min_bin, self.max_bin)
        colors = plt.cm.viridis(np.linspace(0, 1, self.num_channels))

        self.axes = []
        self.containers = []
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
            ax.set_xlim(self.min_bin, self.max_bin)
            ax.set_xlabel("Bin")
            ax.set_ylabel("Photon Counts")

        plt.tight_layout()

    def update(self, frame):
        """
        Updates the histogram data for each frame.

        Parameters:
            frame (int): Current frame number.
        """
        if not plt.fignum_exists(self.fig.number) and self.show:
            get_logger().info("Closing GUI...")
            return
        get_logger().info(f"Frame {frame}")
        histograms = self.sensor.accumulate(1)
        for idx, channel in enumerate(self.channel_mask):
            container = self.containers[idx]
            histogram = histograms[channel, self.min_bin : self.max_bin]
            for rect in container:
                rect.set_height(0)
            for rect, h in zip(container, histogram):
                rect.set_height(h)
        self.adjust_ylim(histograms)
        return list(chain(*self.containers))

    def adjust_ylim(self, histograms):
        """
        Adjusts the Y-axis limits based on the histogram data.

        Parameters:
            histograms (np.ndarray): Histogram data for all channels.
        """
        if self.ylim is not None:
            for ax in self.axes:
                ax.set_ylim(0, self.ylim)
        elif self.autoscale:
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

    def save_animation(self, ani):
        """
        Saves the animation to a file.

        Parameters:
            ani (FuncAnimation): The animation object to save.
        """
        get_logger().info("Saving animation...")
        filename = self.filename
        if not filename.endswith(".mp4"):
            filename += ".mp4"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        get_logger().info(f"Saving animation to {filename}...")
        ani.save(filename, writer="ffmpeg", dpi=400, fps=10)
        get_logger().info("Saved animation.")
        plt.close(self.fig)


# ================


@register
class PyQtGraphDashboard(SPADDashboard):
    """
    Dashboard implementation using PyQtGraph for real-time visualization.
    """

    def run(self):
        """
        Executes the PyQtGraph dashboard application.
        """
        global pg, QtWidgets, QtCore
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        signal.signal(signal.SIGINT, signal.SIG_DFL)

        class DashboardWindow(pg.GraphicsLayoutWidget):
            """
            Custom window class for handling key events.
            """

            def __init__(self, parent=None):
                super().__init__(parent)

            def keyPressEvent(self, event):
                """
                Handles key press events to allow exiting the application.
                """
                if hasattr(QtCore.Qt, "Key_Escape"):
                    # PyQt5
                    escape_list = [QtCore.Qt.Key_Escape, QtCore.Qt.Key_Q]
                else:
                    # PyQt6
                    escape_list = [QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape]
                if event.key() in escape_list:
                    QtWidgets.QApplication.quit()

        app = QtWidgets.QApplication([])
        win = DashboardWindow()
        if self.fullscreen:
            win.showFullScreen()
        else:
            win.show()

        self.setup_plots(win)

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1)

        if self.show:
            app.exec()

    def setup_plots(self, win):
        """
        Sets up the plots for each channel in the dashboard.

        Parameters:
            win (DashboardWindow): The main window for the plots.
        """
        rows = int(np.ceil(np.sqrt(self.num_channels)))
        cols = int(np.ceil(self.num_channels / rows))

        self.plots = []
        self.bars = []
        bins = np.arange(self.min_bin, self.max_bin)
        for idx, channel in enumerate(self.channel_mask):
            row, col = divmod(idx, cols)
            if col == 0:
                win.nextRow()
            p = win.addPlot()
            self.plots.append(p)
            y = np.zeros_like(bins)
            bg = pg.BarGraphItem(x=bins, height=y, width=1.0, brush="b")
            p.addItem(bg)
            self.bars.append(bg)
            p.setLabel("bottom", "Bin")
            p.setLabel("left", "Photon Counts")
            p.setXRange(self.min_bin, self.max_bin, padding=0)

    def update(self):
        """
        Updates the histogram data in the plots.
        """
        histograms = self.sensor.accumulate(1)
        for idx, channel in enumerate(self.channel_mask):
            histogram = histograms[channel, self.min_bin : self.max_bin]
            self.bars[idx].setOpts(height=histogram)
            if self.ylim is not None:
                self.plots[idx].setYRange(0, self.ylim)
            elif self.autoscale:
                self.plots[idx].enableAutoRange(axis="y")
        if not any([plot.isVisible() for plot in self.plots]):
            get_logger().info("Closing GUI...")
            QtWidgets.QApplication.quit()


# ================


@register
class DashDashboard(SPADDashboard):
    """
    Dashboard implementation using Dash and Plotly for web-based visualization.
    """

    def run(self):
        """
        Executes the Dash dashboard application.
        """
        global make_subplots, go, dash, dcc, html, Input, Output, State
        import dash
        import plotly.graph_objs as go
        from dash import dcc, html
        from dash.dependencies import Input, Output, State
        from plotly.subplots import make_subplots

        self.app = dash.Dash(__name__)
        self.histograms = np.zeros((self.num_channels, self.max_bin - self.min_bin))
        self.setup_layout()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run_dash)
        self.thread.start()
        try:
            while self.thread.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            get_logger().info("Closing GUI...")

    def setup_layout(self):
        """
        Sets up the layout and figures for the Dash application.
        """
        self.bins = np.arange(self.min_bin, self.max_bin)
        rows = int(np.ceil(np.sqrt(self.num_channels)))
        cols = int(np.ceil(self.num_channels / rows))

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"Channel {ch}" for ch in self.channel_mask],
            shared_xaxes="all",
            shared_yaxes="all",
        )

        for idx, channel in enumerate(self.channel_mask):
            row = idx // cols + 1
            col = idx % cols + 1
            trace = go.Bar(
                x=self.bins,
                y=self.histograms[idx],
                name=f"Channel {channel}",
                marker_color="blue",
            )
            fig.add_trace(trace, row=row, col=col)
            fig.update_xaxes(title_text="Bin", row=row, col=col, showticklabels=True)
            fig.update_yaxes(
                title_text="Photon Counts", row=row, col=col, showticklabels=True
            )
            fig.update_xaxes(range=[self.min_bin, self.max_bin], row=row, col=col)
            if self.ylim is not None:
                fig.update_yaxes(range=[0, self.ylim], row=row, col=col)

        fig.update_layout(autosize=True, showlegend=False)

        self.app.layout = html.Div(
            [
                dcc.Graph(
                    id="live-update-graph",
                    figure=fig,
                    style={"height": "100vh", "width": "100vw"},
                ),
                dcc.Interval(
                    id="interval-component",
                    interval=0,
                    n_intervals=0,
                    max_intervals=self.num_frames,
                ),
            ],
        )

        self.app.callback(
            Output("live-update-graph", "figure"),
            [Input("interval-component", "n_intervals")],
            [State("live-update-graph", "figure")],
        )(self.update_graph_live)

    def run_dash(self):
        """
        Runs the Dash server in a separate thread.
        """
        self.app.run_server(debug=False, use_reloader=False)

    def update_graph_live(self, n_intervals, existing_fig):
        """
        Updates the live graph with new histogram data.

        Parameters:
            n_intervals (int): The number of intervals that have passed.
            existing_fig (dict): The existing figure to update.
        """
        if n_intervals is None:
            return dash.no_update
        acquired = self.lock.acquire(blocking=False)
        if acquired:
            try:
                histograms = self.sensor.accumulate(1)
            finally:
                self.lock.release()
        else:
            histograms = self.histograms
        self.histograms = histograms

        for idx, channel in enumerate(self.channel_mask):
            histogram = histograms[channel, self.min_bin : self.max_bin]
            existing_fig["data"][idx]["y"] = histogram.tolist()
            yaxis_key = f"yaxis{idx + 1}" if idx > 0 else "yaxis"

            if self.ylim is not None:
                existing_fig["layout"][yaxis_key]["range"] = [0, self.ylim]
            elif self.autoscale:
                existing_fig["layout"][yaxis_key]["autorange"] = True
        return existing_fig
