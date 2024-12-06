"""Dashboard for SPAD sensors.

This module provides a dashboard for visualizing SPAD sensor data in real-time. There
are three implementations available with different supported features:

- :class:`~drivers.spads.dashboard.MatplotlibDashboard`: Uses Matplotlib for
    visualization.
- :class:`~drivers.spads.dashboard.PyQtGraphDashboard`: Uses PyQtGraph for
    visualization.
- :class:`~drivers.spads.dashboard.DashDashboard`: Uses Dash and Plotly for web-based
    visualization.

You can specify user-defined callbacks to be executed on each update of the dashboard.

Example:

.. code-block:: python

    from cc_hardware.drivers.spads import SPADSensor
    from cc_hardware.drivers.spads.dashboard import SPADDashboard

    sensor = SPADSensor.create_from_registry(...)
    dashboard = SPADDashboard.create_from_registry(
        ...,
        sensor=sensor,
        user_callback=my_callback,
    )

    dashboard.run()
"""

import signal
import threading
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path
from typing import Callable, Self

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
        min_bin (int, optional): Minimum bin value for histogram.
        max_bin (int, optional): Maximum bin value for histogram.
        autoscale (bool): Whether to autoscale the histogram. Default is True.
        ylim (float, optional): Y-axis limit for the histogram.
        channel_mask (list[int], optional): List of channels to display.
        user_callback (Callable[[Self], None], optional): User-defined callback
            function. It should accept the dashboard instance as an argument.
    """

    def __init__(
        self,
        sensor: SPADSensor,
        num_frames: int = 1_000_000,
        min_bin: int | None = None,
        max_bin: int | None = None,
        autoscale: bool = True,
        ylim: float | None = None,
        channel_mask: list[int] | None = None,
        user_callback: Callable[[Self], None] | None = None,
    ):
        self.sensor = sensor
        self.num_frames = num_frames
        self._min_bin = min_bin
        self._max_bin = max_bin
        self.autoscale = autoscale
        self.ylim = ylim
        self.channel_mask = channel_mask
        self.user_callback = user_callback

        if self.autoscale and self.ylim is not None:
            get_logger().warning(
                "Autoscale is enabled, but ylim is set. Disabling autoscale."
            )
            self.autoscale = False

        self.setup_sensor()
        get_logger().info("Starting histogram GUI...")

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
    def run(
        self,
        *,
        fullscreen: bool = False,
        headless: bool = False,
        save: Path | None = None,
    ):
        """
        Abstract method to display the dashboard.

        Parameters:
            fullscreen (bool): Whether to display in fullscreen mode.
            headless (bool): Whether to run in headless mode.
            save (Path | None): If provided, save the output to this file.
        """
        pass

    # ================

    @property
    def min_bin(self) -> int:
        """
        Minimum bin value for the histogram.

        Supports variable sized bins based on the sensor configuration.
        """
        if self._min_bin is None:
            return 0
        return self._min_bin

    @property
    def max_bin(self) -> int:
        """
        Maximum bin value for the histogram.

        Supports variable sized bins based on the sensor configuration.
        """
        if self._max_bin is None:
            return self.sensor.num_bins
        return self._max_bin


# ================


@register
class MatplotlibDashboard(SPADDashboard):
    """
    Dashboard implementation using Matplotlib for visualization.
    """

    def run(
        self,
        *,
        fullscreen: bool = False,
        headless: bool = False,
        save: Path | None = None,
    ):
        """
        Executes the Matplotlib dashboard with real-time updates.

        Parameters:
            fullscreen (bool): Whether to display in fullscreen mode.
            headless (bool): Whether to run in headless mode (without GUI).
            save (Path | None): If provided, save the output to this file.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        from cc_hardware.utils.plotting import set_matplotlib_style

        signal.signal(signal.SIGINT, signal.SIG_DFL)

        set_matplotlib_style()

        self.fullscreen_mode = fullscreen
        self.save_path = save
        self.headless = headless

        self.setup_plot()

        self.ani = FuncAnimation(
            self.fig,
            self.update,
            frames=range(self.num_frames),
            interval=1,
            repeat=False,
            blit=True,
        )

        if self.save_path:
            self.save_animation(self.ani, self.save_path)

        if not self.headless:
            plt.show()

    def setup_plot(self):
        """
        Sets up the Matplotlib plot layout and styling.
        """
        import matplotlib.pyplot as plt

        rows = int(np.ceil(np.sqrt(self.num_channels)))
        cols = int(np.ceil(self.num_channels / rows))

        self.fig = plt.figure(figsize=(6, 6 * rows / cols))
        self.gs = plt.GridSpec(rows, cols, figure=self.fig)

        if self.fullscreen_mode:
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
        import matplotlib.pyplot as plt

        if not plt.fignum_exists(self.fig.number):
            get_logger().info("Closing GUI...")
            return

        # Check if the user has updated the number of bins
        if self.sensor.num_bins != self.bins.size:
            # Update x-axis
            self.bins = np.arange(self.sensor.num_bins + 1)  # Adjust bin range
            for container in self.containers:
                for rect, x in zip(container, self.bins[:-1]):
                    rect.set_x(x)  # Update x-coordinates of histogram bars
            for ax in self.axes:
                ax.set_xlim(0, self.sensor.num_bins)  # Update x-axis limits
                ax.set_xticks(np.linspace(0, self.sensor.num_bins, 5))  # Update x-ticks

        histograms = self.sensor.accumulate(1)
        self.adjust_ylim(histograms)

        for idx, channel in enumerate(self.channel_mask):
            container = self.containers[idx]
            histogram = histograms[channel, self.min_bin : self.max_bin]
            for rect, h in zip(container, histogram):
                rect.set_height(h)

        # Call user callback if provided
        if self.user_callback is not None:
            self.user_callback(self)

        # Force a background flush to update the plot

        return list(chain(*self.containers))

    def adjust_ylim(self, histograms: np.ndarray):
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

    def save_animation(self, ani, filename: str):
        """
        Saves the animation to a file.

        Parameters:
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


# ================


@register
class PyQtGraphDashboard(SPADDashboard):
    """
    Dashboard implementation using PyQtGraph for real-time visualization.
    """

    def run(
        self,
        *,
        fullscreen: bool = False,
        headless: bool = False,
        save: Path | None = None,
    ):
        """
        Executes the PyQtGraph dashboard application.

        Parameters:
            fullscreen (bool): Whether to display in fullscreen mode.
            headless (bool): Whether to run in headless mode.
            save (Path | None): If provided, save the output to this file.
        """
        if headless:
            raise NotImplementedError(
                "Headless mode is not supported for PyQtGraphDashboard."
            )
        if save:
            raise NotImplementedError(
                "Save functionality is not implemented for PyQtGraphDashboard."
            )

        global pg, QtWidgets, QtCore
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        signal.signal(signal.SIGINT, signal.SIG_DFL)

        class DashboardWindow(QtWidgets.QWidget):
            """
            Custom window class with a fixed settings panel on the right.
            """

            def __init__(self, parent=None):
                super().__init__(parent)
                self.init_ui()

            def init_ui(self):
                """
                Initializes the user interface with a settings panel and plots.
                """
                # Main horizontal splitter
                self.splitter = QtWidgets.QSplitter(
                    QtCore.Qt.Orientation.Horizontal, self
                )

                # Left: PyQtGraph area
                self.graphic_view = pg.GraphicsLayoutWidget()
                self.splitter.addWidget(self.graphic_view)

                # Right: Settings panel
                self.settings_panel = QtWidgets.QWidget()
                self.settings_layout = QtWidgets.QVBoxLayout(self.settings_panel)
                self.splitter.addWidget(self.settings_panel)

                # Add settings widgets
                self.autoscale_checkbox = QtWidgets.QCheckBox("Autoscale")
                self.autoscale_checkbox.setChecked(True)  # Default value
                self.settings_layout.addWidget(self.autoscale_checkbox)

                self.shared_y_checkbox = QtWidgets.QCheckBox("Shared Y-Axis")
                self.shared_y_checkbox.setChecked(True)  # Default value
                self.settings_layout.addWidget(self.shared_y_checkbox)

                self.log_y_checkbox = QtWidgets.QCheckBox("Log Y-Axis")
                self.log_y_checkbox.setChecked(False)  # Default value
                self.settings_layout.addWidget(self.log_y_checkbox)

                self.y_limit_textbox = QtWidgets.QLineEdit()
                self.y_limit_textbox.setPlaceholderText("Enter Y-Limit")
                self.y_limit_textbox.setEnabled(not self.autoscale_checkbox.isChecked())
                self.settings_layout.addWidget(QtWidgets.QLabel("Y-Limit"))
                self.settings_layout.addWidget(self.y_limit_textbox)

                self.settings_layout.addStretch()  # Add spacer to align widgets at top

                # Set proportions
                self.splitter.setStretchFactor(0, 4)  # Graphics view takes more space
                self.splitter.setStretchFactor(1, 1)  # Settings panel takes less space

                # Main layout
                layout = QtWidgets.QVBoxLayout(self)
                layout.addWidget(self.splitter)

            def keyPressEvent(self, event):
                """
                Handles key press events to allow exiting the application.
                """
                if event.key() in [QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape]:
                    QtWidgets.QApplication.quit()

        app = QtWidgets.QApplication([])
        win = DashboardWindow()
        self.win = win
        if fullscreen:
            win.showFullScreen()
        else:
            win.show()

        self.shared_y = True

        self.setup_plots(win.graphic_view)

        # Connect settings to functionality
        win.autoscale_checkbox.stateChanged.connect(self.toggle_autoscale)
        win.shared_y_checkbox.stateChanged.connect(self.toggle_shared_y)
        win.y_limit_textbox.textChanged.connect(self.update_y_limit)
        win.log_y_checkbox.stateChanged.connect(self.toggle_log_y)

        win.autoscale_checkbox.setChecked(self.autoscale)
        win.shared_y_checkbox.setChecked(self.shared_y)
        if self.ylim is not None:
            win.y_limit_textbox.setText(str(self.ylim))

        self.toggle_autoscale(self.autoscale)
        self.toggle_shared_y(self.shared_y)

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1)

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
            bg = self._create_bar_graph_item(bins, y)
            p.addItem(bg)
            self.bars.append(bg)
            p.setLabel("bottom", "Bin")
            p.setLabel("left", "Photon Counts")
            p.setXRange(self.min_bin, self.max_bin, padding=0)

            if not self.autoscale:
                p.enableAutoRange(axis="y", enable=False)

    def update(self):
        """
        Updates the histogram data in the plots.
        """
        histograms = np.array(self.sensor.accumulate(1))

        # If log scale is enabled, replace 0s with 1s to avoid log(0)
        ymin = 0
        if self.win.log_y_checkbox.isChecked():
            histograms = np.where(histograms < 1, 1, histograms)
            ymin = 1

        ylim = None
        if self.win.y_limit_textbox.isEnabled():
            ylim = self.ylim
        if self.autoscale and self.shared_y:
            # Set ylim to be max of _all_ channels
            ylim = int(histograms[:, self.min_bin : self.max_bin].max()) + 1

        for idx, channel in enumerate(self.channel_mask):
            histogram = histograms[channel, self.min_bin : self.max_bin]

            try:
                self.bars[idx].setOpts(height=histogram)
            except ValueError:
                get_logger().warning("Histogram size has changed.")

                # Histogram size has changed.
                # Remove the old BarGraphItem if the shape has changed
                self.plots[idx].removeItem(self.bars[idx])

                # Create a new BarGraphItem and add it to the plot
                x = np.arange(self.min_bin, self.max_bin)
                if len(histogram) < len(x):
                    histogram = np.pad(histogram, (0, len(x) - len(histogram)))
                self.bars[idx] = self._create_bar_graph_item(x)
                self.plots[idx].addItem(self.bars[idx])
                self.plots[idx].setXRange(self.min_bin, self.max_bin)

            channel_ylim = ylim
            if self.autoscale and not self.shared_y:
                channel_ylim = histogram.max() + 1
            if channel_ylim is not None:
                self.plots[idx].setLimits(yMin=ymin, yMax=channel_ylim)
                self.plots[idx].setYRange(ymin, channel_ylim)

        # Call user callback if provided
        if self.user_callback is not None:
            self.user_callback(self)

        if not any([plot.isVisible() for plot in self.plots]):
            get_logger().info("Closing GUI...")
            QtWidgets.QApplication.quit()

    def _create_bar_graph_item(self, bins, y=None):
        import pyqtgraph as pg

        y = np.zeros_like(bins) if y is None else y
        return pg.BarGraphItem(x=bins, height=y, width=1.0, brush="b")

    def toggle_autoscale(self, state: int):
        get_logger().debug(f"Autoscale: {bool(state)}")
        self.autoscale = bool(state)

        self.win.y_limit_textbox.setEnabled(not self.win.autoscale_checkbox.isChecked())
        if self.autoscale:
            self.win.y_limit_textbox.clear()

    def toggle_shared_y(self, state: int):
        get_logger().debug(f"Shared Y-Axis: {bool(state)}")
        self.shared_y = bool(state)

    def toggle_log_y(self, state: int):
        get_logger().debug(f"Log Y-Axis: {bool(state)}")
        for plot in self.plots:
            plot.setLogMode(y=bool(state))

    def update_y_limit(self):
        text = self.win.y_limit_textbox.text()
        if text.isdigit():
            self.ylim = int(text)
            get_logger().debug(f"Y-Limit set to: {self.ylim}")
            for plot in self.plots:
                plot.setYRange(0, self.ylim)
        else:
            get_logger().debug("Invalid Y-Limit input")


# ================


@register
class DashDashboard(SPADDashboard):
    """
    Dashboard implementation using Dash and Plotly for web-based visualization.
    """

    def run(
        self,
        *,
        fullscreen: bool = False,
        headless: bool = False,
        save: Path | None = None,
    ):
        """
        Executes the Dash dashboard application.

        Parameters:
            fullscreen (bool): Unused parameter for DashDashboard.
            headless (bool): Whether to run in headless mode.
            save (Path | None): If provided, save the dashboard to this file.
        """
        if fullscreen:
            get_logger().warning(
                "Fullscreen functionality is not applicable for DashDashboard."
            )
        if save:
            raise NotImplementedError(
                "Save functionality is not implemented for DashDashboard."
            )

        global make_subplots, go, dash, dcc, html, Input, Output, State
        import dash
        import plotly.graph_objs as go
        from dash import dcc, html
        from dash.dependencies import Input, Output, State
        from plotly.subplots import make_subplots

        self.app = dash.Dash(__name__)
        self.histograms = np.zeros((self.num_channels, self.max_bin - self.min_bin))
        self.num_updates = 0
        self.setup_layout()
        self.lock = threading.Lock()

        self.app.run(debug=False, use_reloader=False)

    def setup_layout(self):
        """
        Sets up the layout and figures for the Dash application.
        """
        import dash
        import plotly.graph_objs as go
        from dash import dcc, html
        from plotly.subplots import make_subplots

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
                    interval=1,
                    n_intervals=0,
                    max_intervals=self.num_frames,
                ),
            ],
        )

        @self.app.callback(
            Output("live-update-graph", "figure"),
            [Input("interval-component", "n_intervals")],
            [State("live-update-graph", "figure")],
        )
        def update_graph_live(n_intervals, existing_fig):
            """
            Updates the live graph with new histogram data.

            Parameters:
                n_intervals (int): The number of intervals that have passed.
                existing_fig (dict): The existing figure to update.
            """
            self.num_updates += 1
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
                xaxis_key = f"xaxis{idx + 1}" if idx > 0 else "xaxis"
                yaxis_key = f"yaxis{idx + 1}" if idx > 0 else "yaxis"

                bins = np.arange(self.min_bin, self.max_bin)

                # Update x and y for each channel
                existing_fig["data"][idx]["y"] = histogram.tolist()
                if len(existing_fig["data"][idx]["x"]) != len(bins):
                    existing_fig["data"][idx]["x"] = bins.tolist()
                    existing_fig["layout"][xaxis_key]["range"] = [
                        self.min_bin,
                        self.max_bin,
                    ]

                if self.ylim is not None:
                    existing_fig["layout"][yaxis_key]["range"] = [0, self.ylim]
                elif self.autoscale:
                    existing_fig["layout"][yaxis_key]["autorange"] = True

            # Call user callback if provided
            if self.user_callback is not None:
                self.user_callback(self)

            return existing_fig
