"""Dashboards for SPAD sensors."""

from cc_hardware.drivers.spads import SPADSensor, SPADSensorConfig
from cc_hardware.tools.dashboard.spad_dashboard.spad_dashboard import (
    SPADDashboard,
    SPADDashboardConfig,
)
from cc_hardware.utils import Manager, register_cli, run_cli

# =============================================================================
# Register the dashboard implementations

SPADDashboard.register("PyQtGraphDashboard", f"{__name__}.pyqtgraph")
SPADDashboardConfig.register("PyQtGraphDashboardConfig", f"{__name__}.pyqtgraph")
SPADDashboardConfig.register(
    "PyQtGraphDashboardConfig", f"{__name__}.pyqtgraph", "PyQtGraphDashboard"
)

SPADDashboard.register("MatplotlibDashboard", f"{__name__}.matplotlib")
SPADDashboardConfig.register("MatplotlibDashboardConfig", f"{__name__}.matplotlib")
SPADDashboardConfig.register(
    "MatplotlibDashboardConfig", f"{__name__}.matplotlib", "MatplotlibDashboard"
)

SPADDashboard.register("DashDashboard", f"{__name__}.dash")
SPADDashboardConfig.register("DashDashboardConfig", f"{__name__}.dash")
SPADDashboardConfig.register("DashDashboardConfig", f"{__name__}.dash", "DashDashboard")

# =============================================================================


@register_cli
def spad_dashboard(sensor: SPADSensorConfig, dashboard: SPADDashboardConfig):
    def setup(manager: Manager):
        """Configures the manager with sensor and dashboard instances.

        Args:
            manager (Manager): Manager to add sensor and dashboard to.
        """
        _sensor = SPADSensor.create_from_config(sensor)
        manager.add(sensor=_sensor)

        _dashboard = SPADDashboard.create_from_config(config=dashboard, sensor=_sensor)
        _dashboard.setup()
        manager.add(dashboard=_dashboard)

    def loop(
        frame: int, manager: Manager, sensor: SPADSensor, dashboard: SPADDashboard
    ):
        """Updates dashboard each frame.

        Args:
            frame (int): Current frame number.
            manager (Manager): Manager controlling the loop.
            sensor (SPADSensor): Sensor instance (unused here).
            dashboard (SPADDashboard): Dashboard instance to update.
        """
        dashboard.update(frame)

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


def main():
    run_cli(spad_dashboard)


# =============================================================================

__all__ = [
    "SPADDashboard",
    "SPADDashboardConfig",
]
