from functools import partial
from typing import Any

from cc_hardware.drivers.sensor import Sensor
from cc_hardware.tools.app import APP
from cc_hardware.utils.logger import get_logger

# ========================


def calibrate(sensor: type[Sensor] | Sensor, **kwargs) -> Any:
    from cc_hardware.utils.manager import Manager

    calibration_data = None

    def setup(manager: Manager, sensor: Sensor):
        nonlocal calibration_data
        calibration_data = sensor.calibrate()

    with Manager(sensor=sensor) as manager:
        manager.run(setup=setup)

    return calibration_data


@APP.command()
def tmf8828_calibrate(filename: str, port: str | None = None):
    """This method will perform four calibrations, one for the TMF8828 mode and one for
    the legacy TMF882x mode, and each twice with each range mode."""
    from cc_hardware.drivers.spads.tmf8828 import SPADID, RangeMode, TMF8828Sensor

    TMF8828Sensor.PORT = port or TMF8828Sensor.PORT
    spad_ids = [SPADID.ID6, SPADID.ID15]
    range_modes = [RangeMode.SHORT, RangeMode.LONG]
    data: list[str] = []
    for spad_id in spad_ids:
        for range_mode in range_modes:
            get_logger().info(f"Calibrating SPAD {spad_id} in {range_mode} range mode.")
            data.extend(
                calibrate(
                    partial(TMF8828Sensor, spad_id=spad_id, range_mode=range_mode)
                )
            )

    # Write the calibration data to a file
    with open(filename, "w") as f:
        f.write("\n".join(data))
