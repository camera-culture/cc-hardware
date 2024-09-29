import os
from pathlib import Path

from cc_hardware.tools.app import APP
from cc_hardware.utils.logger import get_logger


@APP.command()
def upload(port: str, script: Path):
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


@APP.command()
def tmf8828_upload():
    from cc_hardware.drivers.spads.tmf8828 import TMF8828Sensor

    get_logger().info(
        f"Uploading TMF8828 sensor sketch from {TMF8828Sensor.SCRIPT} "
        f"to port {TMF8828Sensor.PORT}"
    )
    upload(port=TMF8828Sensor.PORT, script=TMF8828Sensor.SCRIPT)
