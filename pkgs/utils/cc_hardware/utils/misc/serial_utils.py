"""Utilities for working with serial devices."""

import os
from pathlib import Path
from typing import Type

import psutil
from serial import Serial
from serial.serialutil import SerialException
from serial.tools import list_ports

from cc_hardware.utils.logger import get_logger


def find_device_by_label(label: str) -> Path | None:
    """Find a device by its volume label.

    Args:
        label: The volume label to search for.
    """

    get_logger().info(f"Searching for device with label {label}...")
    for part in psutil.disk_partitions(all=False):
        try:
            # Check if the volume label matches
            if Path(part.device).resolve().name == label or label in part.mountpoint:
                get_logger().info(
                    f"Found device with label {label} at {part.mountpoint}"
                )
                return part.mountpoint
        except PermissionError:
            # Skip any devices that raise a permission error
            continue

    get_logger().warning(f"Device with label {label} not found")
    return None


def find_ports(cls: Type[Serial] | None = None, /, **kwargs) -> list[Serial | str]:
    """Check all available ports for a device.

    Args:
        cls: The serial class to use. If None, only the port names are returned.
        **kwargs: Additional keyword arguments to pass to the serial class.

    Returns:
        A list of serial ports or port names.
    """
    get_logger().info("Opening all potential serial ports...")
    serial_ports = []
    the_ports_list = list_ports.comports()
    for port in the_ports_list:
        if port.pid is None:
            continue
        try:
            if cls is None:
                serial_port = port.device
            else:
                serial_port = cls(port.device, **kwargs)
        except SerialException:
            continue
        serial_ports.append(serial_port)
        get_logger().info(f"\t{port.device}")

    return serial_ports


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
