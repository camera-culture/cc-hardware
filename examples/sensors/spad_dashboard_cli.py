import time
from functools import partial

from cc_hardware.drivers.spads import SPADSensor, SPADSensorConfig
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli

i = 0
t0 = 0


def my_callback(dashboard: SPADDashboard):
    """Calls logger at intervals.

    Args:
        dashboard (SPADDashboard): The dashboard instance to use in the callback.
    """
    global i
    i += 1
    if i % 10 == 0:
        get_logger().info("Callback called")


def setup(manager: Manager, sensor: SPADSensorConfig, dashboard: SPADDashboardConfig):
    """Configures the manager with sensor and dashboard instances.

    Args:
        manager (Manager): Manager to add sensor and dashboard to.
    """
    sensor: SPADSensor = SPADSensor.create_from_config(sensor)
    manager.add(sensor=sensor)

    dashboard.user_callback = my_callback
    dashboard: SPADDashboard = dashboard.create_from_registry(
        config=dashboard, sensor=sensor
    )
    dashboard.setup()
    manager.add(dashboard=dashboard)


def loop(frame: int, manager: Manager, sensor: SPADSensor, dashboard: SPADDashboard):
    """Updates dashboard each frame.

    Args:
        frame (int): Current frame number.
        manager (Manager): Manager controlling the loop.
        sensor (SPADSensor): Sensor instance (unused here).
        dashboard (SPADDashboard): Dashboard instance to update.
    """
    global t0

    if frame % 10 == 0:
        t1 = time.time()
        fps = 10 / (t1 - t0)
        t0 = time.time()
        get_logger().info(f"Frame: {frame}, FPS: {fps:.2f}")

    histograms = sensor.accumulate()
    dashboard.update(frame, histograms=histograms)


@register_cli
def spad_dashboard_demo(sensor: SPADSensorConfig, dashboard: SPADDashboardConfig):
    """Sets up and runs the SPAD dashboard.

    Args:
        sensor (SPADSensorConfig): Configuration object for the sensor.
        dashboard (SPADDashboardConfig): Configuration object for the dashboard.
    """

    global t0
    t0 = time.time()

    with Manager() as manager:
        manager.run(setup=partial(setup, sensor=sensor, dashboard=dashboard), loop=loop)


if __name__ == "__main__":
    run_cli(spad_dashboard_demo)
