"""Tools for working with serial components."""

import os
from pathlib import Path

from cc_hardware.tools.app import APP, typer
from cc_hardware.utils.logger import get_logger

# ========================

serial_tools_APP = typer.Typer()
APP.add_typer(serial_tools_APP, name="serial")

# ========================


@serial_tools_APP.command()
def arduino_upload(port: str, script: Path):
    """Upload an Arduino sketch to the given port."""

    # Check arduino-cli is installed
    if os.system("arduino-cli version") != 0:
        raise RuntimeError("arduino-cli is not installed")

    # Check the port exists
    if not os.path.exists(port):
        raise FileNotFoundError(f"Port {port} does not exist")

    # Run the upload command
    assert script.exists(), f"Script {script} does not exist"
    cmd = f"arduino-cli compile --upload --port {port} --fqbn arduino:avr:uno {script}"
    if os.system(cmd) != 0:
        raise RuntimeError("Failed to upload the sketch")


@serial_tools_APP.command()
def tmf8828_upload(port: str, script: Path | None = None):
    """Upload the TMF8828 sensor sketch to the given port.

    Uses the TMF8828Sensor.SCRIPT attribute to locate the sketch if none is provided.
    """

    from cc_hardware.drivers.spads.tmf8828 import TMF8828Sensor

    script = script or TMF8828Sensor.SCRIPT
    get_logger().info(f"Uploading TMF8828 sensor sketch from {script} to port {port}")
    arduino_upload(port=port, script=script)


@serial_tools_APP.command()
def vl53l8ch_upload(
    port: str, script: Path | None = None, *, build: bool = True, verbose: bool = False
):
    """Upload the VL53L8CH sensor sketch to the given port.

    Uses the VL53L8CHSensor.SCRIPT attribute to locate the sketch if none is provided.
    """
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
