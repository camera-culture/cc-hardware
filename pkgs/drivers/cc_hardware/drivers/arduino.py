import threading
import time
from functools import singledispatchmethod
from typing import Any, Self

import serial
from serial.serialutil import SerialException
from serial.tools import list_ports

from cc_hardware.utils.logger import get_logger


class Arduino(serial.Serial):
    _lock = threading.Lock()

    @classmethod
    def create(
        cls, port: str | None = None, *, wait: int = 2, **kwargs
    ) -> Self | list[Self]:
        """Create an Arduino object from a serial port. Will check all available ports
        (if port is None) and wait for the Arduino to reset before connecting."""
        serial_ports = []
        if port is None:
            get_logger().info("Opening all potential serial ports...")
            the_ports_list = list_ports.comports()
            for port in the_ports_list:
                if port.pid is None:
                    continue
                try:
                    serial_port = cls(port.device, **kwargs)
                except SerialException:
                    continue
                serial_ports.append(serial_port)
                get_logger().info(f"\t{port.device}")
        else:
            serial_ports = [cls(port, **kwargs)]

        assert len(serial_ports) > 0, f"No {cls.__name__} devices found"

        get_logger().info(
            f"Waiting {wait} seconds for {cls.__name__} devices to reset..."
        )
        time.sleep(wait)

        return serial_ports if len(serial_ports) > 1 else serial_ports[0]

    @singledispatchmethod
    def write(self, data: Any):
        get_logger().warning(f"Invalid data type: {type(data)}")
        with self._lock:  # Ensure that only one thread can write at a time
            super().write(data)

    @write.register
    def _(self, data: str):
        with self._lock:  # Ensure that only one thread can write at a time
            super().write(bytes(data, "utf-8"))

    def read(self, size: int = 1) -> bytes:
        with self._lock:  # Ensure that only one thread can read at a time
            return super().read(size)
