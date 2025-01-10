"""Dashboard for SPAD sensors.

This module provides a dashboard for visualizing SPAD sensor data in real-time. There
are three implementations available with different supported features:

- :class:`~cc_hardware.drivers.spads.dashboards.matplotlib.MatplotlibDashboard`:
    Uses Matplotlib for visualization.
- :class:`~cc_hardware.drivers.spads.dashboards.pyqtgraph.PyQtGraphDashboard`:
    Uses PyQtGraph for visualization.
- :class:`~cc_hardware.drivers.spads.dashboards.dash.DashDashboard`:
    Uses Dash and Plotly for web-based visualization.

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
from typing import Callable, Self
from functools import cached_property

import numpy as np

from cc_hardware.drivers.spads.spad import SPADSensor
from cc_hardware.utils import Registry, get_logger
from cc_hardware.utils.config import CCHardwareConfig, config_wrapper


@config_wrapper
class SPADDashboardConfig(CCHardwareConfig):
    """
    Configuration for SPAD sensor dashboards.

    When defining a new dashboard, create a subclass of this configuration class and add
    any necessary parameters.

    Attributes:
        num_frames (int): Number of frames to process. Default is 1,000,000.
        min_bin (int, optional): Minimum bin value for histogram.
        max_bin (int, optional): Maximum bin value for histogram.
        autoscale (bool): Whether to autoscale the histogram. Default is True.
        ylim (float, optional): Y-axis limit for the histogram.
        channel_mask (list[int], optional): List of channels to display.
        user_callback (Callable[[Self], None], optional): User-defined callback
            function. It should accept the dashboard instance as an argument.
    """

    instance: str = "SPADDashboard"

    num_frames: int = 1_000_000
    min_bin: int | None = None
    max_bin: int | None = None
    autoscale: bool = True
    ylim: float | None = None
    channel_mask: list[int] | None = None
    user_callback: Callable[[Self], None] | None = None


class SPADDashboard(ABC, Registry):
    """
    Abstract base class for SPAD sensor dashboards.

    Args:
        config (SPADDashboardConfig): The dashboard configuration
        sensor (SPADSensor): The SPAD sensor instance.
    """

    def __init__(
        self,
        config: SPADDashboardConfig,
        sensor: SPADSensor,
        resolution: tuple[int, int] | None = None,
    ):
        self._config = config
        self._sensor = sensor

        self.num_channels: int
        self.channel_mask: np.ndarray

        if self.config.autoscale and self.config.ylim is not None:
            get_logger().warning(
                "Autoscale is enabled, but ylim is set. Disabling autoscale."
            )
            self.config.autoscale = False

        self._setup_sensor(resolution)
        get_logger().info("Starting histogram GUI...")

    def _setup_sensor(self, resolution: tuple[int, int] | None = None):
        """
        Configures the sensor settings and channel mask.
        """
        h, w = resolution or self._sensor.resolution
        total_channels = h * w
        self.channel_mask = np.arange(total_channels)
        if self.config.channel_mask is not None:
            self.channel_mask = np.array(self.config.channel_mask)
        self.num_channels = len(self.channel_mask)
        get_logger().debug(f"Setup sensor with {self.num_channels} channels.")

    @property
    def config(self) -> SPADDashboardConfig:
        """Retrieves the dashboard configuration."""
        return self._config

    @property
    def sensor(self) -> SPADSensor:
        """Retrieves the SPAD sensor instance."""
        return self._sensor

    @abstractmethod
    def setup(self):
        """
        Abstract method to set up the dashboard. Should be independent of whether the
        dashboard is run in a loop or not.
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
        if self.config.min_bin is None:
            return 0
        return self.config.min_bin

    @property
    def max_bin(self) -> int:
        """
        Maximum bin value for the histogram.

        Supports variable sized bins based on the sensor configuration.
        """
        if self.config.max_bin is None:
            return self._sensor.num_bins
        return self.config.max_bin
