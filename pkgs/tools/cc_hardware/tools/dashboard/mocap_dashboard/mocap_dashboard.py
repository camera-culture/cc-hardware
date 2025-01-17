"""Dashboard for motion capture sensors."""

import numpy as np

from cc_hardware.drivers.mocap import MotionCaptureSensor
from cc_hardware.tools.dashboard import DashboardConfig, Dashboard
from cc_hardware.utils import config_wrapper, get_logger


@config_wrapper
class MotionCaptureDashboardConfig(DashboardConfig):
    """
    Configuration for motion capture dashboards.

    When defining a new dashboard, create a subclass of this configuration class and add
    any necessary parameters.
    """

    pass


class MotionCaptureDashboard[T: MotionCaptureDashboardConfig](Dashboard[T]):
    """
    Abstract base class for MotionCapture sensor dashboards.

    Args:
        config (MotionCaptureDashboardConfig): The dashboard configuration
        sensor (MotionCaptureSensor): The MotionCapture sensor instance.
    """

    def __init__(
        self,
        config: T,
        sensor: MotionCaptureSensor,
    ):
        super().__init__(config)
        self._sensor = sensor

    @property
    def sensor(self) -> MotionCaptureSensor:
        """Retrieves the MotionCapture sensor instance."""
        return self._sensor
