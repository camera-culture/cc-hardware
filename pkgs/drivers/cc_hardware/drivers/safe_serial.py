"""SafeSerial class for thread-safe serial communication.

The :class:`~drivers.safe_serial.SafeSerial` class is a wrapper around the
`serial.Serial \
    <https://pyserial.readthedocs.io/en/latest/pyserial_api.html#serial.Serial>`_
class that provides a thread-safe interface for reading and writing to a serial device.
It also provides a few convenience methods for reading and writing data.
"""

import re
import threading
import time
from functools import singledispatchmethod
from typing import Any, Self

import serial
from serial.serialutil import SerialException
from serial.tools import list_ports

from cc_hardware.utils.logger import get_logger


class SafeSerial(serial.Serial):
    """
    A thread-safe implementation of the serial.Serial class that synchronizes read and
    write operations using a lock. Provides additional utility methods for creating
    instances and handling data writes in different formats.

    Args:
        *args: Positional arguments passed to the parent serial.Serial class.
        **kwargs: Keyword arguments passed to the parent serial.Serial class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()
        self.flush()

    @classmethod
    def create(
        cls, port: str | None = None, *, wait: int = 2, **kwargs
    ) -> Self | list[Self]:
        """
        Create an instance of SafeSerial from a serial port. Checks all available ports
        if no port is specified and waits for the serial device to reset before
        establishing the connection.

        Args:
            port (str | None): The specific serial port to connect to, or None to
                check all available ports.
            wait (int): The number of seconds to wait for the device to reset. Default
                is 2 seconds.
            **kwargs: Additional keyword arguments passed to the SafeSerial constructor.

        Returns:
            SafeSerial | list[SafeSerial]: A single instance if a port is specified
                or multiple instances if no port is specified.
        """
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
        """
        Write data to the serial port. If the data type is invalid, a warning is logged.

        Args:
            data (Any): The data to write to the serial port.

        Returns:
            None
        """
        get_logger().warning(f"Invalid data type: {type(data)}")
        with self._lock:
            super().write(data)

    @write.register
    def _(self, data: str):
        """
        Write a string to the serial port by encoding it as UTF-8 bytes.

        Args:
            data (str): The string to write to the serial port.

        Returns:
            None
        """
        with self._lock:
            super().write(bytes(data, "utf-8"))

    def read(self, size: int = 1) -> bytes:
        """
        Read a specified number of bytes from the serial port.

        Args:
            size (int): The number of bytes to read. Default is 1.

        Returns:
            bytes: The bytes read from the serial port.
        """
        with self._lock:
            return super().read(size)

    def wait_for_start_talk(self, timeout: float = None) -> bytes | None:
        """
        Wait until SafeSerial starts talking. Returns data if successful,
        None if timeout.

        Args:
            timeout (float | None): Maximum time to wait before giving up. Defaults to
                None.

        Returns:
            bytes | None: The received data if SafeSerial starts talking, otherwise
                None.
        """
        data = b""
        start_time = time.time()
        while len(data) == 0:
            if timeout is not None and time.time() - start_time > timeout:
                return None
            data = self.read()
        return data

    def wait_for_stop_talk(self, timeout: float = None) -> bytes | None:
        """
        Wait until SafeSerial stops talking. Returns accumulated data if stopped before
            timeout.

        Args:
            timeout (float | None): Maximum time to wait before giving up. Defaults to
                None.

        Returns:
            bytes | None: The accumulated data if SafeSerial stops talking, otherwise
                None.
        """
        data = b"0"
        accumulated_data = b""
        start_time = time.time()
        while len(data) > 0:
            if timeout is not None and time.time() - start_time > timeout:
                return None
            data = self.read()
            try:
                data_str = re.sub(r"[\r\n]", "", data.decode("utf-8").strip())
                get_logger().debug(data_str)
            except UnicodeDecodeError:
                get_logger().debug(data)
            accumulated_data += data
        return accumulated_data

    def write_and_wait_for_start_talk(
        self, data: str, timeout: float | None = None, tries: int = 10
    ) -> bool:
        """
        Write data to SafeSerial and wait for it to start talking with timeout.
        If timeout happens before something is received, resend data.

        Args:
            data (str): The data to be written.
            timeout (float | None): The maximum wait time for each attempt. Defaults to
                instance timeout.
            tries (int): The number of attempts to perform. Defaults to 10.

        Returns:
            bool: True if successful, False otherwise.
        """
        for attempt in range(tries):
            get_logger().debug(
                f"Attempt {attempt + 1}/{tries}: Writing '{data}' "
                "and waiting for SafeSerial to start talking."
            )
            self.write(data)
            received_data = self.wait_for_start_talk(timeout)
            if received_data is not None:
                get_logger().debug("SafeSerial started talking.")
                return True
            get_logger().warning(
                "Timeout occurred waiting for SafeSerial to start talking. "
                "Retrying..."
            )
        return False

    def write_and_wait_for_stop_talk(
        self,
        data: str,
        timeout: float | None = None,
        tries: int = 10,
        return_data: bool = False,
    ) -> bool | tuple[bool, bytes | None]:
        """
        Write data to SafeSerial and wait for it to stop talking with timeout.
        If timeout happens before something is received, resend data.

        Args:
            data (str): The data to be written.
            timeout (float | None): The maximum wait time for each attempt. Defaults to
                instance timeout.
            tries (int): The number of attempts to perform. Defaults to 10.
            return_data (bool): Whether to return the accumulated data upon success.
                Defaults to False.

        Returns:
            bool | tuple[bool, bytes | None]: True if successful, otherwise False.
                If return_data is True, returns a tuple of success status and
                accumulated data.
        """
        for attempt in range(tries):
            get_logger().debug(
                f"Attempt {attempt + 1}/{tries}: Writing '{data}' and "
                "waiting for SafeSerial to start talking."
            )
            self.write(data)
            received_data = self.wait_for_start_talk(timeout)
            if received_data is None:
                get_logger().warning(
                    "Timeout occurred waiting for SafeSerial to start "
                    "talking. Retrying..."
                )
                continue
            received_data += self.wait_for_stop_talk(timeout)
            if received_data is not None:
                get_logger().debug("SafeSerial stopped talking.")
                return (True, received_data) if return_data else True
        return (False, None) if return_data else False

    def write_and_wait_for_start_and_stop_talk(
        self, data: str, timeout: float | None = None, tries: int = 10
    ) -> bool:
        """
        Write data to SafeSerial and wait for it to start and stop talking with timeout.
        If timeout happens before either event, resend data.

        Args:
            data (str): The data to be written.
            timeout (float | None): The maximum wait time for each attempt. Defaults to
                instance timeout.
            tries (int): The number of attempts to perform. Defaults to 10.

        Returns:
            bool: True if successful, False otherwise.
        """
        for attempt in range(tries):
            get_logger().debug(
                f"Attempt {attempt + 1}/{tries}: Writing '{data}' and "
                "waiting for SafeSerial to start and stop talking."
            )
            self.write(data)
            if not self.wait_for_start_talk(timeout):
                get_logger().warning(
                    "Timeout occurred waiting for SafeSerial to start "
                    "talking. Retrying..."
                )
                continue
            if self.wait_for_stop_talk(timeout):
                get_logger().debug("SafeSerial stopped talking.")
                return True
        return False
