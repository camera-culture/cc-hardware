import os
from pathlib import Path

from cc_hardware.drivers.spads.vl53l8ch import VL53L8CHSensor
from cc_hardware.tools.cli import register_cli, run_cli
from cc_hardware.utils import get_logger
from cc_hardware.utils.serial_utils import find_device_by_label


@register_cli
def spad_upload(
    port: str | None = None,
    script: Path | None = None,
    build: bool = True,
    verbose: bool = False,
):
    if port is None:
        # Attempt to find the port
        # Will be something like "NOD_F401RE"
        port = find_device_by_label("NOD_F401RE")
        assert port is not None, "Could not find VL53L8CH device"

    assert Path(port).exists(), f"Port {port} does not exist"

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


if __name__ == "__main__":
    run_cli(spad_upload)
