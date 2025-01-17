from cc_hardware.drivers.spads import SPADSensor
from cc_hardware.drivers.spads.spad_wrappers import SPADWrapperConfig
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils import Manager, register_cli, run_cli


@register_cli
def spad_dashboard(sensor: SPADWrapperConfig, dashboard: SPADDashboardConfig):
    """Sets up and runs the SPAD dashboard.

    Args:
        sensor (SPADSensorConfig): Configuration object for the sensor.
        dashboard (SPADDashboardConfig): Configuration object for the dashboard.
    """

    def setup(manager: Manager):
        """Configures the manager with sensor and dashboard instances.

        Args:
            manager (Manager): Manager to add sensor and dashboard to.
        """
        _sensor: SPADSensor = SPADSensor.create_from_config(sensor)
        manager.add(sensor=_sensor)

        _dashboard: SPADDashboard = dashboard.create_from_registry(
            config=dashboard, sensor=_sensor
        )
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
        histograms = sensor.accumulate()
        dashboard.update(frame, histograms=histograms)

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


if __name__ == "__main__":
    run_cli(spad_dashboard)
