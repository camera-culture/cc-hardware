"""SPAD dashboard based on PyQtGraph for real-time visualization."""

import signal
from functools import partial
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from cc_hardware.drivers.spads import SPADDashboard
from cc_hardware.utils import get_logger


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
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)

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


class PyQtGraphDashboard(SPADDashboard):
    """
    Dashboard implementation using PyQtGraph for real-time visualization.
    """

    def setup(
        self,
        *,
        fullscreen: bool = False,
        headless: bool = False,
        save: Path | None = None,
    ):
        """
        Sets up the PyQtGraph plot layout and styling.

        Args:
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

        # Reset signal handler to default to allow closing the application
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        self.app = QtWidgets.QApplication([])
        win = DashboardWindow()
        self.win = win
        if fullscreen:
            win.showFullScreen()
        else:
            win.show()

        self.shared_y = True

        rows = int(np.ceil(np.sqrt(self.num_channels)))
        cols = int(np.ceil(self.num_channels / rows))

        self.plots = []
        self.bars = []
        bins = np.arange(self.min_bin, self.max_bin)

        for idx in range(len(self.channel_mask)):
            _, col = divmod(idx, cols)
            if col == 0:
                win.graphic_view.nextRow()
            p: pg.PlotItem = win.graphic_view.addPlot()
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

        # Connect settings to functionality
        self.win.autoscale_checkbox.stateChanged.connect(self.toggle_autoscale)
        self.win.shared_y_checkbox.stateChanged.connect(self.toggle_shared_y)
        self.win.y_limit_textbox.textChanged.connect(self.update_y_limit)
        self.win.log_y_checkbox.stateChanged.connect(self.toggle_log_y)

        self.win.autoscale_checkbox.setChecked(self.autoscale)
        self.win.shared_y_checkbox.setChecked(self.shared_y)
        if self.ylim is not None:
            self.win.y_limit_textbox.setText(str(self.ylim))

        self.toggle_autoscale(self.autoscale)
        self.toggle_shared_y(self.shared_y)

    def run(self):
        """
        Executes the PyQtGraph dashboard application.

        Args:
            fullscreen (bool): Whether to display in fullscreen mode.
            headless (bool): Whether to run in headless mode.
            save (Path | None): If provided, save the output to this file.
        """

        global pg, QtWidgets, QtCore
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(partial(self.update, frame=-1, step=False))
        self.timer.start(1)

        self.app.exec()

    def update(
        self,
        frame: int,
        *,
        histograms: np.ndarray | None = None,
        step: bool = True,
    ):
        """
        Updates the histogram data in the plots.
        """
        if histograms is None:
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

        if step:
            self.app.processEvents()

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
