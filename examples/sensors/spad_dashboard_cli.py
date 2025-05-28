import time
from datetime import datetime
from functools import partial
from pathlib import Path

from cc_hardware.drivers.spads import SPADDataType, SPADSensor, SPADSensorConfig
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers import PklHandler

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


def setup(
    manager: Manager,
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    record: bool = False,
):
    """Configures the manager with sensor and dashboard instances."""

    assert (
        SPADDataType.HISTOGRAM in sensor.data_type
    ), "Sensor must support histogram data type."
    _sensor: SPADSensor = SPADSensor.create_from_config(sensor)
    manager.add(sensor=_sensor)

    dashboard.user_callback = my_callback
    _dashboard: SPADDashboard = dashboard.create_from_registry(
        config=dashboard, sensor=_sensor
    )
    _dashboard.setup()
    manager.add(dashboard=_dashboard)

    if record:
        now = datetime.now()
        logdir = Path("logs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
        logdir.mkdir(parents=True, exist_ok=True)
        get_logger().info(f"Logging to {logdir}")
        pkl_handler = PklHandler(logdir / "data.pkl")
        manager.add(pkl_handler=pkl_handler)
        pkl_handler.append({"config": sensor.to_dict()})


def loop(
    frame: int,
    manager: Manager,
    sensor: SPADSensor,
    dashboard: SPADDashboard,
    pkl_handler: PklHandler | None = None,
):
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

    data = sensor.accumulate()
    dashboard.update(frame, data=data)

    if pkl_handler is not None:
        pkl_handler.append({"frame": frame, **data})


@register_cli
def spad_dashboard_demo(
    sensor: SPADSensorConfig, dashboard: SPADDashboardConfig, record: bool = False
):
    """Sets up and runs the SPAD dashboard.

    Args:
        sensor (SPADSensorConfig): Configuration object for the sensor.
        dashboard (SPADDashboardConfig): Configuration object for the dashboard.
    """

    global t0
    t0 = time.time()

    with Manager() as manager:
        manager.run(
            setup=partial(setup, sensor=sensor, dashboard=dashboard, record=record),
            loop=loop,
        )


if __name__ == "__main__":
    run_cli(spad_dashboard_demo)
