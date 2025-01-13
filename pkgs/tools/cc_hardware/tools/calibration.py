"""Tools for calibrating hardware devices."""

from functools import partial
from typing import Any

from cc_hardware.drivers.sensor import Sensor
from cc_hardware.drivers.spads.tmf8828 import SPADID, RangeMode, TMF8828Sensor
from cc_hardware.utils import Manager, get_logger, register_cli


def calibrate(sensor: type[Sensor] | Sensor, **kwargs) -> Any:
    """Calibrate the given sensor and return the calibration data.

    Args:
        sensor (type[Sensor] | Sensor): The sensor to calibrate.

    Keyword Args:
        **kwargs: Additional keyword arguments to pass to the sensor constructor.

    Returns:
        Any: The calibration data. Specific to the sensor implementation.
    """

    calibration_data = None

    def setup(manager: Manager, sensor: Sensor):
        nonlocal calibration_data
        calibration_data = sensor.calibrate()

    with Manager(sensor=sensor) as manager:
        manager.run(setup=setup)

    return calibration_data


@register_cli
def tmf8828_calibrate(
    filename: str,
    port: str | None = None,
    spad_ids: list[SPADID] = [SPADID.ID6, SPADID.ID15],
    range_modes: list[RangeMode] = [RangeMode.SHORT, RangeMode.LONG],
):
    """Calibrate the TMF8828 sensor and save the calibration data to a file.

    This method will perform four calibrations, one for the TMF8828 mode and one for
    the legacy TMF882x mode, and each twice with each range mode."""

    data: list[str] = []
    for spad_id in spad_ids:
        for range_mode in range_modes:
            get_logger().info(f"Calibrating SPAD {spad_id} in {range_mode} range mode.")
            sensor = partial(
                TMF8828Sensor,
                port=port,
                spad_id=spad_id,
                range_mode=range_mode,
            )
            data.extend(calibrate(sensor))

    # Write the calibration data to a file
    with open(filename, "w") as f:
        f.write("\n".join(data))
