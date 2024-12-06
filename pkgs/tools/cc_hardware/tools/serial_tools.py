"""Tools for working with serial components."""

import os
from pathlib import Path

from cc_hardware.tools.app import APP, typer
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.serial_utils import find_device_by_label, find_ports

# ========================

serial_tools_APP = typer.Typer()
APP.add_typer(serial_tools_APP, name="serial")

# ========================


@serial_tools_APP.command()
def arduino_upload(port: str | None, script: Path):
    """Upload an Arduino sketch to the given port."""

    # Check arduino-cli is installed
    if os.system("arduino-cli version") != 0:
        raise RuntimeError("arduino-cli is not installed")

    # Check the port exists
    if port is None:
        serial_ports = find_ports()
        assert len(serial_ports) == 1, "Multiple serial ports found, please specify one"
        port = serial_ports[0]
    if not os.path.exists(port):
        raise FileNotFoundError(f"Port {port} does not exist")

    # Run the upload command
    assert script.exists(), f"Script {script} does not exist"
    cmd = f"arduino-cli compile --upload --port {port} --fqbn arduino:avr:uno {script}"
    if os.system(cmd) != 0:
        raise RuntimeError("Failed to upload the sketch")


@serial_tools_APP.command()
def tmf8828_upload(port: str | None = None, script: Path | None = None):
    """Upload the TMF8828 sensor sketch to the given port.

    Uses the TMF8828Sensor.SCRIPT attribute to locate the sketch if none is provided.
    """

    from cc_hardware.drivers.spads.tmf8828 import TMF8828Sensor

    script = script or TMF8828Sensor.SCRIPT
    get_logger().info(f"Uploading TMF8828 sensor sketch from {script} to port {port}")
    arduino_upload(port=port, script=script)


@serial_tools_APP.command()
def vl53l8ch_upload(
    port: str | None = None,
    script: Path | None = None,
    *,
    build: bool = True,
    verbose: bool = False,
):
    """Upload the VL53L8CH sensor sketch to the given port.

    Uses the VL53L8CHSensor.SCRIPT attribute to locate the sketch if none is provided.

    Args:
        port: The port to upload the sketch to. This is the device path as a storage
            device, not the /dev/tty* path (e.g. /media/username/DEVICE_NAME).
        script: The path to the sketch file. Defaults to the VL53L8CHSensor.SCRIPT
            attribute.

    Keyword Args:
        build: Whether to build the sketch before uploading. Defaults to True.
        verbose: Whether to show the build output. Defaults to False.
    """
    if port is None:
        # Attempt to find the port
        # Will be something like "NOD_F401RE"
        port = find_device_by_label("NOD_F401RE")
        assert port is not None, "Could not find VL53L8CH device"

    assert Path(port).exists(), f"Port {port} does not exist"

    from cc_hardware.drivers.spads.vl53l8ch import VL53L8CHSensor

    script = script or VL53L8CHSensor.SCRIPT

    if build:
        get_logger().info(f"Building VL53L8CH sensor sketch from {script}")
        cmd = f"make -C {script.parent} clean all {'-s' if not verbose else ''}"
        if os.system(cmd) != 0:
            raise RuntimeError("Failed to build the sketch")

    get_logger().info(f"Uploading VL53L8CH sensor sketch from {script} to port {port}")
    cmd = f"make -C {script.parent} upload PORT={port}"
    if os.system(cmd) != 0:
        raise RuntimeError("Failed to upload the sketch")
