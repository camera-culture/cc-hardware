"""SPAD dashboard based on PyQtGraph for real-time visualization."""

import time
from functools import partial

import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.opengl import GLAxisItem, GLScatterPlotItem, GLViewWidget
from pyqtgraph.Qt import QtCore, QtWidgets

from cc_hardware.drivers.spads import SPADDataType
from cc_hardware.tools.dashboard.spad_dashboard import (
    SPADDashboard,
    SPADDashboardConfig,
)
from cc_hardware.utils import config_wrapper, get_logger
from cc_hardware.utils.setting import BoolSetting, OptionSetting, RangeSetting, Setting


@config_wrapper
class PyQtGraphDashboardConfig(SPADDashboardConfig):
    """
    Configuration for the PyQtGraph dashboard.
    """

    fullscreen: bool = False


class DashboardWindow(QtWidgets.QWidget):
    """
    Custom window class with a fixed settings panel on the right.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

    def init_ui(self, settings: dict[str, Setting]):
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

        for name, setting in settings.items():
            title = setting.title or name.replace("_", " ").title()
            if isinstance(setting, RangeSetting):
                setting_layout = QtWidgets.QHBoxLayout()
                title_label = QtWidgets.QLabel(title)
                setting_layout.addWidget(title_label)
                widget = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                widget.setRange(setting.min, setting.max)
                widget.setValue(setting.value)
                widget.setTickInterval(max((setting.max - setting.min) // 20, 1))
                widget.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
                widget.valueChanged.connect(lambda v, s=setting: s.update(v))
                setting_layout.addWidget(widget)
                value_label = QtWidgets.QLabel(str(setting.value))
                widget.valueChanged.connect(
                    lambda v, label=value_label: label.setText(str(v))
                )
                setting_layout.addWidget(value_label)
                self.settings_layout.addLayout(setting_layout)
            elif isinstance(setting, OptionSetting):
                widget = QtWidgets.QComboBox()
                widget.addItems([str(o) for o in setting.options])
                widget.setCurrentIndex(setting.options.index(setting.value))
                widget.currentIndexChanged.connect(
                    lambda v, s=setting: s.update(s.options[v])
                )
                self.settings_layout.addWidget(QtWidgets.QLabel(title))
                self.settings_layout.addWidget(widget)
            elif isinstance(setting, BoolSetting):
                widget = QtWidgets.QCheckBox(title)
                widget.setChecked(setting.value)
                widget.stateChanged.connect(lambda v, s=setting: s.update(bool(v)))
                self.settings_layout.addWidget(widget)

        self.settings_layout.addStretch()  # Add spacer to align widgets at top

        # Set proportions
        self.splitter.setStretchFactor(0, 4)  # Graphics view takes more space
        self.splitter.setStretchFactor(1, 1)  # Settings panel takes less space

        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.splitter)

        # after building self.settings_layout
        self.last_time = time.time()
        self.fps = 0.0
        self.fps_label = QtWidgets.QLabel("FPS: 0.0")
        self.settings_layout.addWidget(self.fps_label)

    def enable_point_cloud_view(self) -> None:
        """Creates a resizable 3-D pane under the histogram grid."""
        if hasattr(self, "pc_view"):
            return  # already added

        if not hasattr(self, "left_splitter"):
            self.left_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            self.graphic_view.setParent(None)
            self.left_splitter.addWidget(self.graphic_view)
            self.splitter.insertWidget(0, self.left_splitter)

        self.pc_view = GLViewWidget()
        self.left_splitter.addWidget(self.pc_view)

        # Replace graphic_view in the main splitter with the new splitter
        self.splitter.insertWidget(0, self.left_splitter)

        # Handle‑sizing ratios (histograms : point‑cloud)
        self.left_splitter.setStretchFactor(0, 3)
        self.left_splitter.setStretchFactor(1, 2)

        # Reset split
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)

    def enable_depth_view(self) -> None:
        """Adds a depth‑image pane under any existing plots."""
        if hasattr(self, "depth_view"):
            return

        if not hasattr(self, "left_splitter"):
            self.left_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            self.graphic_view.setParent(None)
            self.left_splitter.addWidget(self.graphic_view)
            self.splitter.insertWidget(0, self.left_splitter)

        # depth view
        self.depth_viewbox = pg.ViewBox(lockAspect=True, invertY=False)
        self.depth_view = pg.GraphicsLayoutWidget()
        self.depth_view.addItem(self.depth_viewbox)
        self.left_splitter.addWidget(self.depth_view)

        # stretch factors: hist : pc (if any) : depth
        for i in range(self.left_splitter.count()):
            self.left_splitter.setStretchFactor(i, 2)

    def keyPressEvent(self, event):
        """
        Handles key press events to allow exiting the application.
        """
        if event.key() in [QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape]:
            QtWidgets.QApplication.quit()


class PyQtGraphDashboard(SPADDashboard[PyQtGraphDashboardConfig]):
    """
    Dashboard implementation using PyQtGraph for real-time visualization.
    """

    def setup(self):
        """
        Sets up the PyQtGraph plot layout and styling.

        Args:
            fullscreen (bool): Whether to display in fullscreen mode.
        """

        self.app = QtWidgets.QApplication([])

        self._create_plots()

    def _create_plots(self):
        if hasattr(self, "win"):
            self.win.close()

        self.win = DashboardWindow()
        self.win.init_ui(self.sensor.settings)

        if SPADDataType.POINT_CLOUD in self.sensor.config.data_type:
            self.win.enable_point_cloud_view()
            self._pc_plot = GLScatterPlotItem(size=10.0, color=(0.2, 0.6, 1.0, 1.0))
            self.win.pc_view.addItem(self._pc_plot)
            self._axis = GLAxisItem()  # RGB axes, length = 1 by default
            self.win.pc_view.addItem(self._axis)
            self.win.pc_view.opts["distance"] = 5  # camera back a bit

        if SPADDataType.DISTANCE in self.sensor.config.data_type:
            self.win.enable_depth_view()
            self._depth_img = pg.ImageItem(axisOrder="row-major", smooth=False)
            self.win.depth_viewbox.addItem(self._depth_img)
            # red‑to‑blue LUT (red = near, blue = far)
            lut = np.zeros((256, 4), dtype=np.ubyte)
            lut[:, 0] = np.linspace(255, 0, 256)  # R
            lut[:, 2] = np.linspace(0, 255, 256)  # B
            lut[:, 3] = 255  # alpha
            # lut = np.repeat(np.arange(256, dtype=np.ubyte)[:, None], 4, 1)
            # lut[:, 3] = 255  # opaque
            self._depth_img.setLookupTable(lut)

        if self.config.fullscreen:
            self.win.showFullScreen()
        else:
            self.win.show()

        self.shared_y = True

        cols, rows = self.sensor.resolution

        self.plots = []
        self.bars = []
        bins = np.arange(self.min_bin, self.max_bin)

        for idx in range(len(self.channel_mask)):
            _, col = divmod(idx, cols)
            if col == 0:
                self.win.graphic_view.nextRow()
            p: pg.PlotItem = self.win.graphic_view.addPlot()
            self.plots.append(p)
            y = np.zeros_like(bins)
            bg = self._create_bar_graph_item(bins, y)
            p.addItem(bg)
            self.bars.append(bg)
            p.setLabel("bottom", "Bin")
            p.setLabel("left", "Photon Counts")
            p.setXRange(self.min_bin, self.max_bin, padding=0)

            if not self.config.autoscale:
                p.enableAutoRange(axis="y", enable=False)

        # Connect settings to functionality
        self.win.autoscale_checkbox.stateChanged.connect(self.toggle_autoscale)
        self.win.shared_y_checkbox.stateChanged.connect(self.toggle_shared_y)
        self.win.y_limit_textbox.textChanged.connect(self.update_y_limit)
        self.win.log_y_checkbox.stateChanged.connect(self.toggle_log_y)

        self.win.autoscale_checkbox.setChecked(self.config.autoscale)
        self.win.shared_y_checkbox.setChecked(self.shared_y)
        if self.config.ylim is not None:
            self.win.y_limit_textbox.setText(str(self.config.ylim))

        self.toggle_autoscale(self.config.autoscale)
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
        data: dict[SPADDataType, np.ndarray] | None = None,
        step: bool = True,
    ):
        """
        Updates the histogram data in the plots.
        """
        # Update any settings
        self.sensor.update()

        if data is None:
            data = self.sensor.accumulate(1)
        assert SPADDataType.HISTOGRAM in data, "Histogram data is required."
        histograms = data[SPADDataType.HISTOGRAM]

        # Flatten the histograms
        histograms = histograms.reshape(self.num_channels, -1)

        # Check if the number of channels has changed
        if histograms.shape[0] != len(self.plots):
            get_logger().warning(
                "The number of channels has changed from "
                f"{len(self.plots)} to {histograms.shape[0]}."
            )
            self._setup_sensor()
            self._create_plots()
            return

        # If log scale is enabled, replace 0s with 1s to avoid log(0)
        ymin = 0
        if self.win.log_y_checkbox.isChecked():
            histograms = np.where(histograms < 1, 1, histograms)
            ymin = 1

        ylim = None
        if self.win.y_limit_textbox.isEnabled():
            ylim = self.config.ylim
        if self.config.autoscale and self.shared_y:
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
            if self.config.autoscale and not self.shared_y:
                channel_ylim = histogram.max() + 1
            if channel_ylim is not None:
                self.plots[idx].setLimits(yMin=ymin, yMax=channel_ylim)
                self.plots[idx].setYRange(ymin, channel_ylim)

        if SPADDataType.POINT_CLOUD in data:
            pc = data[SPADDataType.POINT_CLOUD]
            self._pc_plot.setData(pos=pc.reshape(-1, 3))

            # center camera on data centroid
            if frame == 0:
                ctr = pc.mean(axis=0)
                self.win.pc_view.setCameraPosition(
                    pos=pg.Vector(ctr[0], ctr[1], ctr[2])
                )

        if SPADDataType.DISTANCE in data:
            d = data[SPADDataType.DISTANCE].astype(np.float32)
            rows, cols = self.sensor.resolution[::-1]
            d_img = d.reshape(rows, cols)
            d_img = cv2.resize(
                d_img, (cols * 3, rows * 3), interpolation=cv2.INTER_CUBIC
            )
            self._depth_img.setImage(d_img, levels=(d.min(), d.max()), autoLevels=False)

            # keep view centred & scaled
            self.win.depth_viewbox.autoRange()

        now = time.time()
        dt = now - self.win.last_time
        if dt > 0:
            self.win.fps = 1.0 / dt
            self.win.fps_label.setText(f"FPS: {self.win.fps:.1f}")
        self.win.last_time = now

        # Call user callback if provided
        if self.config.user_callback is not None:
            self.config.user_callback(self)

        if not any([plot.isVisible() for plot in self.plots]):
            get_logger().info("Closing GUI...")
            QtWidgets.QApplication.quit()

        if step:
            self.app.processEvents()

    def _create_bar_graph_item(self, bins, y=None):
        import pyqtgraph as pg

        y = np.zeros_like(bins) if y is None else y
        return pg.BarGraphItem(x=bins + 0.5, height=y, width=1.0, brush="b")

    def toggle_autoscale(self, state: int):
        get_logger().debug(f"Autoscale: {bool(state)}")
        self.config.autoscale = bool(state)

        self.win.y_limit_textbox.setEnabled(not self.win.autoscale_checkbox.isChecked())
        if self.config.autoscale:
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
            self.config.ylim = int(text)
            get_logger().debug(f"Y-Limit set to: {self.config.ylim}")
            for plot in self.plots:
                plot.setYRange(0, self.config.ylim)
        else:
            get_logger().debug("Invalid Y-Limit input")

    @property
    def is_okay(self) -> bool:
        return not self.win.isHidden()

    def close(self):
        QtWidgets.QApplication.quit()
        if hasattr(self, "win") and self.win is not None:
            self.win.close()
        if hasattr(self, "app") and self.app is not None:
            self.app.quit()
            self.app = None
        if hasattr(self, "timer") and self.timer is not None:
            self.timer.stop()
            self.timer = None
