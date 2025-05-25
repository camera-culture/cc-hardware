import time
from datetime import datetime
from functools import partial
from pathlib import Path

from cc_hardware.drivers.spads import SPADDataType, SPADSensor, SPADSensorConfig
from cc_hardware.drivers.spads.vl53l8ch import VL53L8CHConfig8x8
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.tools.dashboard.spad_dashboard.pyqtgraph import (
    PyQtGraphDashboardConfig,
)
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers import PklHandler

# ==========

NOW = datetime.now()
LOGDIR: Path = Path("logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S")
OUTPUT_PKL: Path = LOGDIR / "data.pkl"

SENSOR = VL53L8CHConfig8x8.create(
    num_bins=18,
    subsample=18,
    start_bin=0,
    integration_time_ms=10,
)

DASHBOARD = PyQtGraphDashboardConfig.create(fullscreen=True)
DASHBOARD = None


# ==========


def setup(
    manager: Manager,
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig | None,
    record: bool = False,
):
    """Configures the manager with sensor and dashboard instances.

    Args:
        manager (Manager): Manager to add sensor and dashboard to.
    """
    if record:
        LOGDIR.mkdir(exist_ok=True, parents=True)

        OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
        assert not OUTPUT_PKL.exists(), f"Output file {OUTPUT_PKL} already exists"
        manager.add(writer=PklHandler(OUTPUT_PKL))

    sensor: SPADSensor = SPADSensor.create_from_config(sensor)
    manager.add(sensor=sensor)

    if dashboard is not None:
        dashboard: SPADDashboard = dashboard.create_from_registry(
            config=dashboard, sensor=sensor
        )
        dashboard.setup()
        manager.add(dashboard=dashboard)


def loop(
    frame: int,
    manager: Manager,
    sensor: SPADSensor,
    dashboard: SPADDashboard | None = None,
    writer: PklHandler | None = None,
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
    if dashboard is not None:
        dashboard.update(frame, histograms=data[SPADDataType.HISTOGRAM])

    if writer is not None:
        writer.append({"iter": frame, **data})


@register_cli
def nlos_particle_filter_demo(record: bool = False):
    """Sets up and runs the SPAD dashboard.

    Args:
        sensor (SPADSensorConfig): Configuration object for the sensor.
        dashboard (SPADDashboardConfig): Configuration object for the dashboard.
    """

    global t0
    t0 = time.time()

    with Manager() as manager:
        manager.run(
            setup=partial(setup, record=record, sensor=SENSOR, dashboard=DASHBOARD),
            loop=loop,
        )


if __name__ == "__main__":
    run_cli(nlos_particle_filter_demo)
