from pathlib import Path

from cc_hardware.drivers.spads.tmf8828 import TMF8828Sensor
from cc_hardware.tools.cli import register_cli, run_cli
from cc_hardware.utils import get_logger
from cc_hardware.utils.serial_utils import arduino_upload


@register_cli
def spad_upload(port: str | None = None, script: Path | None = None):
    script = script or TMF8828Sensor.SCRIPT
    get_logger().info(f"Uploading TMF8828 sensor sketch from {script} to port {port}")
    arduino_upload(port=port, script=script)


if __name__ == "__main__":
    run_cli(spad_upload)
