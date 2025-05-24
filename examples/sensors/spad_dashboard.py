import time
from functools import partial
from pathlib import Path
from datetime import datetime

from cc_hardware.drivers.spads import SPADSensor, SPADSensorConfig
from cc_hardware.drivers.spads.vl53l8ch import VL53L8CHConfig4x4
from cc_hardware.drivers.spads.spad_wrappers import SPADMergeWrapperConfig
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.tools.dashboard.spad_dashboard.pyqtgraph import PyQtGraphDashboardConfig
from cc_hardware.utils import Manager, get_logger
from cc_hardware.utils.file_handlers import PklHandler

# ==========

RECORD = False
NOW = datetime.now()
LOGDIR: Path = Path("logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S")
OUTPUT_PKL: Path = LOGDIR / "data.pkl"

# You can start with a config and then change options via create.
WRAPPED_SENSOR = VL53L8CHConfig4x4.create(
    cnh_num_bins=16,
)
SENSOR = SPADMergeWrapperConfig.create(
    wrapped=WRAPPED_SENSOR,
    merge_all=True,
)
DASHBOARD = PyQtGraphDashboardConfig.create(
    fullscreen=True
)

# ==========


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

# ==========


def setup(manager: Manager, sensor: SPADSensorConfig, dashboard: SPADDashboardConfig):
    """Configures the manager with sensor and dashboard instances.

    Args:
        manager (Manager): Manager to add sensor and dashboard to.
    """
    if RECORD:
            LOGDIR.mkdir(exist_ok=True, parents=True)

            OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
            assert not OUTPUT_PKL.exists(), f"Output file {OUTPUT_PKL} already exists"
            manager.add(writer=PklHandler(OUTPUT_PKL))

    sensor: SPADSensor = SPADSensor.create_from_config(sensor)
    manager.add(sensor=sensor)

    dashboard.user_callback = my_callback
    dashboard: SPADDashboard = dashboard.create_from_registry(
        config=dashboard, sensor=sensor
    )
    dashboard.setup()
    manager.add(dashboard=dashboard)


def loop(frame: int, manager: Manager, sensor: SPADSensor, dashboard: SPADDashboard, writer: PklHandler | None = None):
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

    if writer is not None:
        assert writer is not None
        writer.append(
            {
                "iter": frame,
                "histogram": histograms,
            }
        )

if __name__ == "__main__":
    t0 = time.time()

    with Manager() as manager:
        manager.run(setup=partial(setup, sensor=SENSOR, dashboard=DASHBOARD), loop=loop)
