"""Dashboard for SPAD sensors.

This module provides a dashboard for visualizing SPAD sensor data in real-time. There
are three implementations available with different supported features:

- :class:`~cc_hardware.tools.dashboard.spad_dashboard.matplotlib.MatplotlibDashboard`:
    Uses Matplotlib for visualization.
- :class:`~cc_hardware.tools.dashboard.spad_dashboard.pyqtgraph.PyQtGraphDashboard`:
    Uses PyQtGraph for visualization.
- :class:`~cc_hardware.tools.dashboard.spad_dashboard.dash.DashDashboard`:
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

import numpy as np

from cc_hardware.drivers.spads import SPADSensor
from cc_hardware.tools.dashboard import Dashboard, DashboardConfig
from cc_hardware.utils import config_wrapper, get_logger


@config_wrapper
class SPADDashboardConfig(DashboardConfig):
    """
    Configuration for SPAD sensor dashboards.

    When defining a new dashboard, create a subclass of this configuration class and add
    any necessary parameters.

    Attributes:
        min_bin (int, optional): Minimum bin value for histogram.
        max_bin (int, optional): Maximum bin value for histogram.
        autoscale (bool): Whether to autoscale the histogram. Default is True.
        ylim (float, optional): Y-axis limit for the histogram.
        channel_mask (list[int], optional): List of channels to display.
    """

    min_bin: int | None = None
    max_bin: int | None = None
    autoscale: bool = True
    ylim: float | None = None
    channel_mask: list[int] | None = None


class SPADDashboard[T: SPADDashboardConfig](Dashboard[T]):
    """
    Abstract base class for SPAD sensor dashboards.

    Args:
        config (SPADDashboardConfig): The dashboard configuration
        sensor (SPADSensor): The SPAD sensor instance.
    """

    def __init__(
        self,
        config: T,
        sensor: SPADSensor,
    ):
        super().__init__(config)
        self._sensor = sensor

        self.num_channels: int
        self.channel_mask: np.ndarray

        if self.config.autoscale and self.config.ylim is not None:
            get_logger().warning(
                "Autoscale is enabled, but ylim is set. Disabling autoscale."
            )
            self.config.autoscale = False

        self._setup_sensor()
        get_logger().info("Starting histogram GUI...")

    def _setup_sensor(self):
        """
        Configures the sensor settings and channel mask.
        """
        h, w = self._sensor.resolution
        total_channels = h * w
        self.channel_mask = np.arange(total_channels)
        if self.config.channel_mask is not None:
            self.channel_mask = np.array(self.config.channel_mask)
        self.num_channels = len(self.channel_mask)
        get_logger().debug(f"Setup sensor with {self.num_channels} channels.")

    @property
    def sensor(self) -> SPADSensor:
        """Retrieves the SPAD sensor instance."""
        return self._sensor

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
