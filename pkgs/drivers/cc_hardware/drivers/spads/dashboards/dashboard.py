"""Dashboard for SPAD sensors.

This module provides a dashboard for visualizing SPAD sensor data in real-time. There
are three implementations available with different supported features:

- :class:`~drivers.spads.dashboards.matplotlib.MatplotlibDashboard`: Uses Matplotlib for
    visualization.
- :class:`~drivers.spads.dashboards.pyqtgraph.PyQtGraphDashboard`: Uses PyQtGraph for
    visualization.
- :class:`~drivers.spads.dashboards.dash.DashDashboard`: Uses Dash and Plotly for
    web-based visualization.

You can specify user-defined callbacks to be executed on each update of the dashboard.

Example:

.. code-block:: python

    from cc_hardware.drivers.spads import SPADSensor, SPADDashboard

    sensor = SPADSensor.create_from_registry(...)
    dashboard = SPADDashboard.create_from_registry(
        ...,
        sensor=sensor,
        user_callback=my_callback,
    )

    dashboard.run()
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Self

import numpy as np

from cc_hardware.drivers.spads.spad import SPADSensor
from cc_hardware.utils import Registry, get_logger


class SPADDashboard(ABC, Registry):
    """
    Abstract base class for SPAD sensor dashboards.

    Args:
        sensor (SPADSensor): The SPAD sensor instance.

    Keyword Args:
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
        *,
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

        self._setup_sensor()
        get_logger().info("Starting histogram GUI...")

    def _setup_sensor(self):
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
    def setup(
        self,
        *,
        fullscreen: bool = False,
        headless: bool = False,
        save: Path | None = None,
    ):
        """
        Abstract method to set up the dashboard. Should be independent of whether the
        dashboard is run in a loop or not.

        Args:
            fullscreen (bool): Whether to display in fullscreen mode.
            headless (bool): Whether to run in headless mode.
            save (Path | None): If provided, save the output to this file.
        """
        pass

    @abstractmethod
    def run(self):
        """
        Abstract method to display the dashboard. Blocks until the dashboard is closed.
        """
        pass

    def update(self, frame: int, *, histograms: np.ndarray | None = None):
        """
        Abstract method to update the histogram data. This should be capable of being
        used independent of the loop, as in in a main thread and non-blocking.

        Args:
            frame (int): Current frame number.

        Keyword Args:
            histograms (np.ndarray): The histogram data to update. If not provided, the
                sensor will be used to accumulate the histogram data.
        """
        raise NotImplementedError

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
