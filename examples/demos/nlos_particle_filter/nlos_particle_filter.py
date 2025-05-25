import time

from cc_hardware.drivers.spads import SPADSensor, SPADSensorConfig
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils import Manager, register_cli, run_cli


@register_cli
def nlos_particle_filter(
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    sleep: float = 0.1,
):
    def setup(manager: Manager):
        _sensor = SPADSensor.create_from_config(sensor)
        manager.add(sensor=_sensor)

        _dashboard: SPADDashboard = dashboard.create_from_registry(
            config=dashboard, sensor=_sensor
        )
        _dashboard.setup()
        manager.add(dashboard=_dashboard)

    def loop(
        frame: int,
        manager: Manager,
        sensor: SPADSensor,
        dashboard: SPADDashboard,
    ):
        histograms = sensor.accumulate()
        dashboard.update(frame, histograms=histograms)

        time.sleep(sleep)

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


if __name__ == "__main__":
    run_cli(nlos_particle_filter)
