import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from backprojection import (
    BackprojectionAlgorithm,
    BackprojectionConfig,
    BackprojectionDashboard,
    BackprojectionDashboardConfig,
)

from cc_hardware.drivers.spads import SPADDataType, SPADSensor, SPADSensorConfig
from cc_hardware.drivers.spads.pkl import PklSPADSensorConfig
from cc_hardware.drivers.spads.spad_wrappers import (
    SPADBackgroundRemovalWrapperConfig,
    SPADMovingAverageWrapperConfig,
)
from cc_hardware.drivers.spads.vl53l8ch import RangingMode, VL53L8CHConfig4x4
from cc_hardware.tools.dashboard.spad_dashboard import (
    DummySPADDashboardConfig,
    SPADDashboard,
    SPADDashboardConfig,
)
from cc_hardware.tools.dashboard.spad_dashboard.pyqtgraph import (
    PyQtGraphDashboardConfig,
)
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers import PklHandler

# ==========

NOW = datetime.now()
LOGDIR: Path = Path("logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S")
OUTPUT_PKL: Path = LOGDIR / "data.pkl"

WRAPPED_SENSOR = VL53L8CHConfig4x4.create(
    num_bins=18,
    subsample=1,
    start_bin=30,
    ranging_mode=RangingMode.CONTINUOUS,
    ranging_frequency_hz=1,
    data_type=SPADDataType.HISTOGRAM | SPADDataType.POINT_CLOUD | SPADDataType.DISTANCE,
)
WRAPPED_SENSOR = SPADBackgroundRemovalWrapperConfig.create(
    pkl_spad=PklSPADSensorConfig.create(
        pkl_path=Path("logs") / "2025-06-05/11-26-31/data.pkl",
        index=1,
    ),
    wrapped=WRAPPED_SENSOR,
)
WRAPPED_SENSOR = SPADMovingAverageWrapperConfig.create(
    wrapped=WRAPPED_SENSOR,
    window_size=1,
)
SENSOR = WRAPPED_SENSOR

DASHBOARD = PyQtGraphDashboardConfig.create(fullscreen=True)
DASHBOARD = DummySPADDashboardConfig.create()


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
        writer = PklHandler(OUTPUT_PKL)
        manager.add(writer=writer)

        writer.append(dict(config=sensor.to_dict()))

    _sensor: SPADSensor = SPADSensor.create_from_config(sensor)
    manager.add(sensor=_sensor)

    if dashboard is not None:
        dashboard: SPADDashboard = dashboard.create_from_registry(
            config=dashboard, sensor=_sensor
        )
        dashboard.setup()
        manager.add(dashboard=dashboard)

    # Initialize the backprojection algorithm
    backprojection_config = BackprojectionConfig(
        x_range=(-1, 1),
        y_range=(-1, 1),
        z_range=(0, 2),
        num_x=50,
        num_y=50,
        num_z=50,
    )
    backprojection_algorithm = BackprojectionAlgorithm(
        backprojection_config, sensor_config=sensor
    )
    manager.add(algorithm=backprojection_algorithm)

    backprojection_dashboard_config = BackprojectionDashboardConfig(
        xlim=backprojection_config.x_range,
        ylim=backprojection_config.y_range,
        zlim=backprojection_config.z_range,
        xres=backprojection_algorithm.xres,
        yres=backprojection_algorithm.yres,
        zres=backprojection_algorithm.zres,
        num_x=backprojection_config.num_x,
        num_y=backprojection_config.num_y,
        num_z=backprojection_config.num_z,
        # show_
    )
    backprojection_dashboard = BackprojectionDashboard(backprojection_dashboard_config)
    manager.add(backprojection_dashboard=backprojection_dashboard)


def loop(
    frame: int,
    manager: Manager,
    sensor: SPADSensor,
    dashboard: SPADDashboard | None = None,
    writer: PklHandler | None = None,
    algorithm: BackprojectionAlgorithm | None = None,
    backprojection_dashboard: BackprojectionDashboard | None = None,
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
        dashboard.update(frame, data=data)

    if algorithm is not None:
        assert SPADDataType.POINT_CLOUD in data
        assert SPADDataType.HISTOGRAM in data

        volume = algorithm.update(data)

        if backprojection_dashboard is not None:

            def filter_volume(volume: np.ndarray, num_x, num_y) -> np.ndarray:
                volume_unpadded = (
                    2 * volume[:, :, 1:-1] - volume[:, :, :-2] - volume[:, :, 2:]
                )
                zero_pad = np.zeros((num_x, num_y, 1))
                volume_padded = np.concatenate(
                    [zero_pad, volume_unpadded, zero_pad], axis=-1
                )
                return volume_padded

            volume = filter_volume(
                volume,
                num_x=algorithm.config.num_x,
                num_y=algorithm.config.num_y,
            )
            backprojection_dashboard.update(
                volume,
                data[SPADDataType.HISTOGRAM].reshape(
                    -1, data[SPADDataType.HISTOGRAM].shape[-1]
                ),
            )

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
