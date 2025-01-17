"""Dashboards for motion capture sensors."""

from cc_hardware.drivers.mocap import MotionCaptureSensor, MotionCaptureSensorConfig
from cc_hardware.tools.dashboard.mocap_dashboard.mocap_dashboard import (
    MotionCaptureDashboard,
    MotionCaptureDashboardConfig,
)
from cc_hardware.utils import Manager, register_cli, run_cli

# =============================================================================
# Register the dashboard implementations

MotionCaptureDashboard.register(
    "PyQtGraphMotionCaptureDashboard", f"{__name__}.pyqtgraph"
)
MotionCaptureDashboardConfig.register(
    "PyQtGraphMotionCaptureDashboardConfig", f"{__name__}.pyqtgraph"
)
MotionCaptureDashboardConfig.register(
    "PyQtGraphMotionCaptureDashboardConfig",
    f"{__name__}.pyqtgraph",
    "PyQtGraphMotionCaptureDashboard",
)

MotionCaptureDashboard.register("DashMotionCaptureDashboard", f"{__name__}.dash")
MotionCaptureDashboardConfig.register(
    "DashMotionCaptureDashboardConfig", f"{__name__}.dash"
)
MotionCaptureDashboardConfig.register(
    "DashMotionCaptureDashboardConfig", f"{__name__}.dash", "DashMotionCaptureDashboard"
)

# =============================================================================


@register_cli
def mocap_dashboard(
    sensor: MotionCaptureSensorConfig, dashboard: MotionCaptureDashboardConfig
):
    def setup(manager: Manager):
        """Configures the manager with sensor and dashboard instances.

        Args:
            manager (Manager): Manager to add sensor and dashboard to.
        """
        _sensor = MotionCaptureSensor.create_from_config(sensor)
        manager.add(sensor=_sensor)

        _dashboard = MotionCaptureDashboard.create_from_config(
            config=dashboard, sensor=_sensor
        )
        _dashboard.setup()
        manager.add(dashboard=_dashboard)

    def loop(
        frame: int,
        manager: Manager,
        sensor: MotionCaptureSensor,
        dashboard: MotionCaptureDashboard,
    ):
        """Updates dashboard each frame.

        Args:
            frame (int): Current frame number.
            manager (Manager): Manager controlling the loop.
            sensor (MotionCaptureSensor): Sensor instance (unused here).
            dashboard (MotionCaptureDashboard): Dashboard instance to update.
        """
        dashboard.update(frame)

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


def main():
    run_cli(mocap_dashboard)


# =============================================================================

__all__ = [
    "MotionCaptureDashboard",
    "MotionCaptureDashboardConfig",
]
