import pickle
import time

import numpy as np

from cc_hardware.drivers.spads import (
    SPADDashboard,
    SPADDashboardConfig,
    SPADSensor,
    SPADSensorConfig,
)
from cc_hardware.tools.cli import register_cli, run_cli
from cc_hardware.utils import get_logger
from cc_hardware.utils.manager import Manager

i = 0


def my_callback(dashboard: SPADDashboard):
    """Calls logger at intervals.

    Args:
        dashboard (SPADDashboard): The dashboard instance to use in the callback.
    """
    global i
    i += 1
    if i % 10 == 0:
        get_logger().info("Callback called")


@register_cli
def spad_dashboard(
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    *,
    capture_ambient: bool = False,
    merge: bool = False,
):
    """Sets up and runs the SPAD dashboard.

    Args:
        sensor (SPADSensorConfig): Configuration object for the sensor.
        dashboard (SPADDashboardConfig): Configuration object for the dashboard.
    """
    ambient: np.ndarray | None = None

    def setup(manager: Manager):
        """Configures the manager with sensor and dashboard instances.

        Args:
            manager (Manager): Manager to add sensor and dashboard to.
        """
        _sensor = sensor.create_instance()
        manager.add(sensor=_sensor)

        nonlocal ambient
        if capture_ambient:
            get_logger().info("Capturing ambient light...")
            histograms = _sensor.accumulate(20, average=False)
            ambient = np.max(histograms, axis=0)
            get_logger().info("Ambient light captured")
            pickle.dump(ambient, open("ambient.pkl", "wb"))
        else:
            try:
                ambient = pickle.load(open("ambient.pkl", "rb"))
                if ambient.shape[0] != np.prod(_sensor.resolution):
                    get_logger().warning(
                        f"Ambient light data shape {ambient.shape} does not match "
                        f"sensor resolution {np.prod(_sensor.resolution)}"
                    )
                    ambient = None
            except FileNotFoundError:
                ambient = None

        resolution = _sensor.resolution
        if merge:
            resolution = (1, 1)
        dashboard.user_callback = my_callback
        _dashboard: SPADDashboard = dashboard.create_instance(sensor=_sensor, resolution=resolution)
        _dashboard.setup()
        manager.add(dashboard=_dashboard)

    def loop(
        frame: int, manager: Manager, sensor: SPADSensor, dashboard: SPADDashboard
    ) -> bool:
        """Updates dashboard each frame.

        Args:
            frame (int): Current frame number.
            manager (Manager): Manager controlling the loop.
            sensor (SPADSensor): Sensor instance (unused here).
            dashboard (SPADDashboard): Dashboard instance to update.

        Returns:
            bool: Whether to continue running.
        """
        histograms = sensor.accumulate()
        if len(histograms.shape) == 3:
            histograms = histograms.reshape(-1, histograms.shape[-1])
        if ambient is not None:
            assert (
                ambient.shape == histograms.shape
            ), f"{ambient.shape} != {histograms.shape}"
            histograms -= ambient
            histograms = np.clip(histograms, 0, None)
        dashboard.update(frame, histograms=histograms)

        return True

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


if __name__ == "__main__":
    import argparse
    import sys
    from functools import partial

    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--capture-ambient",
    #     action="store_true",
    #     help="Capture ambient light for calibration",
    # )

    # parsed_args, unparsed_args = parser.parse_known_args()

    # sys.argv[1:] = unparsed_args

    run_cli(spad_dashboard)
    # run_cli(partial(spad_dashboard, capture_ambient=parsed_args.capture_ambient))
