"""
Module for VL53L8CH Sensor Driver.

This module provides classes and functions to interface with the VL53L8CH
time-of-flight sensor. It includes configurations, data processing, and sensor
management functionalities necessary for operating the sensor within the
CC Hardware framework.
"""

import multiprocessing
import multiprocessing.synchronize
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pkg_resources

from cc_hardware.drivers.safe_serial import SafeSerial
from cc_hardware.drivers.sensor import SensorData
from cc_hardware.drivers.spads.spad import SPADSensor
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.registry import register

# ===============


@dataclass(kw_only=True)
class SensorConfig:
    """
    Configuration parameters for the VL53L8CH sensor.

    Attributes:
        resolution (int): Sensor resolution (uint16_t).
        ranging_mode (int): Ranging mode (uint16_t).
        ranging_frequency_hz (int): Ranging frequency in Hz (uint16_t).
        integration_time_ms (int): Integration time in milliseconds (uint16_t).
        cnh_start_bin (int): CNH start bin (uint16_t).
        cnh_num_bins (int): Number of CNH bins (uint16_t).
        cnh_subsample (int): CNH subsample rate (uint16_t).
        agg_start_x (int): Aggregation start X coordinate (uint16_t).
        agg_start_y (int): Aggregation start Y coordinate (uint16_t).
        agg_merge_x (int): Aggregation merge X parameter (uint16_t).
        agg_merge_y (int): Aggregation merge Y parameter (uint16_t).
        agg_cols (int): Number of aggregation columns (uint16_t).
        agg_rows (int): Number of aggregation rows (uint16_t).
    """

    resolution: int  # uint16_t
    ranging_mode: int  # uint16_t
    ranging_frequency_hz: int  # uint16_t
    integration_time_ms: int  # uint16_t
    cnh_start_bin: int  # uint16_t
    cnh_num_bins: int  # uint16_t
    cnh_subsample: int  # uint16_t
    agg_start_x: int  # uint16_t
    agg_start_y: int  # uint16_t
    agg_merge_x: int  # uint16_t
    agg_merge_y: int  # uint16_t
    agg_cols: int  # uint16_t
    agg_rows: int  # uint16_t

    def pack(self) -> bytes:
        """
        Packs the sensor configuration into a byte structure.

        Returns:
            bytes: Packed configuration data.
        """
        return struct.pack(
            "<13H",
            self.resolution,
            self.ranging_mode,
            self.ranging_frequency_hz,
            self.integration_time_ms,
            self.cnh_start_bin,
            self.cnh_num_bins,
            self.cnh_subsample,
            self.agg_start_x,
            self.agg_start_y,
            self.agg_merge_x,
            self.agg_merge_y,
            self.agg_cols,
            self.agg_rows,
        )


@dataclass(kw_only=True)
class SensorConfigShared(SensorConfig):
    """
    Shared sensor configuration with default settings.

    Inherits from SensorConfig and provides default values for common parameters.
    """

    ranging_mode: int = 3  # 1 = Continuous, 3 = Autonomous
    ranging_frequency_hz: int = 60
    integration_time_ms: int = 500
    cnh_start_bin: int = 0
    cnh_subsample: int = 16
    agg_start_x: int = 0
    agg_start_y: int = 0
    agg_merge_x: int = 1
    agg_merge_y: int = 1


@dataclass(kw_only=True)
class SensorConfig4x4(SensorConfigShared):
    """
    Sensor configuration for a 4x4 resolution.

    Inherits from SensorConfigShared and sets resolution and aggregation grid size.
    """

    resolution: int = 16
    cnh_num_bins: int = 8
    agg_cols: int = 4
    agg_rows: int = 4


@dataclass(kw_only=True)
class SensorConfig8x8(SensorConfigShared):
    """
    Sensor configuration for an 8x8 resolution.

    Inherits from SensorConfigShared and sets resolution and aggregation grid size.
    """

    resolution: int = 64
    cnh_num_bins: int = 8
    agg_cols: int = 8
    agg_rows: int = 8


# ===============


class VL53L8CHHistogram(SensorData):
    """
    Processes and stores histogram data from the VL53L8CH sensor.

    This class handles the accumulation and processing of histogram data
    received from the sensor, managing multiple pixel histograms.
    """

    def __init__(self):
        """
        Initializes the VL53L8CHHistogram instance.
        """
        super().__init__()

        self._pixel_histograms = {}
        self._has_data = False
        self._num_pixels = None

    def reset(self, num_pixels: int | None = None) -> None:
        """
        Resets the histogram data.

        Args:
            num_pixels (int): Number of pixels expected in the histogram data.
        """
        self._has_data = False
        self._pixel_histograms = {}
        if num_pixels is not None:
            self._num_pixels = num_pixels

    def process(self, row: list[str]) -> bool:
        """
        Processes a row of histogram data.

        Args:
            row (list[str]): A list of string values representing a row of data.

        Returns:
            bool: True if processing is successful, False otherwise.
        """
        assert self._num_pixels is not None, "Number of pixels not set"

        try:
            agg_num = int(row[1])
            if agg_num in self._pixel_histograms:
                get_logger().error(f"Duplicate aggregation number received: {agg_num}")
                return False
        except ValueError:
            get_logger().error("Invalid data formatting received.")
            return False

        try:
            ambient = float(row[3])
            bin_vals = [float(val) - ambient for val in row[5:]]
            self._pixel_histograms[agg_num] = np.clip(bin_vals, 0, None)
        except (ValueError, IndexError):
            get_logger().error("Invalid data formatting received.")
            return False

        if len(self._pixel_histograms) == self._num_pixels:
            self._data = np.array(
                [self._pixel_histograms[k] for k in sorted(self._pixel_histograms)]
            )
            self._pixel_histograms = {}
            self._has_data = True

        return True

    def get_data(self) -> np.ndarray:
        """
        Retrieves the processed histogram data.

        Returns:
            np.ndarray: A copy of the histogram data.
        """
        data = np.copy(self._data)
        self.reset()
        return data

    @property
    def has_data(self) -> bool:
        """
        Checks if histogram data is available.

        Returns:
            bool: True if data is available, False otherwise.
        """
        return self._has_data


# ===============


@register
class VL53L8CHSensor(SPADSensor):
    """
    Main sensor class for the VL53L8CH time-of-flight sensor.

    This class handles communication with the sensor, configuration,
    data acquisition, and data processing.

    Attributes:
        SCRIPT (Path): Path to the sensor's makefile script.
        BAUDRATE (int): Serial communication baud rate.
    """

    SCRIPT: Path = Path(
        pkg_resources.resource_filename(
            "cc_hardware.drivers", str(Path("data") / "vl53l8ch" / "build" / "makefile")
        )
    )
    BAUDRATE: int = 921_600

    def __init__(
        self,
        *,
        port: str | None = None,
        config: SensorConfig = SensorConfig4x4(),
        **kwargs,
    ):
        """
        Initializes the VL53L8CHSensor instance.

        Keyword Args:
            port (str | None): Serial port to which the sensor is connected.
                If None, a port is automatically detected.
            **kwargs: Configuration parameters to update. Keys must match
                the fields of SensorConfig.
        """
        self._config = config
        self._histogram = VL53L8CHHistogram()

        # Use Queue for inter-process communication
        self._queue = multiprocessing.Queue(maxsize=self._config.resolution * 4)
        self._write_queue = multiprocessing.Queue(maxsize=10)
        self._initialized_event = multiprocessing.Event()
        self._stop_event = multiprocessing.Event()

        # Send the configuration to the sensor
        self.update(**kwargs)

        # Start the reader process
        self._reader_process = multiprocessing.Process(
            target=self._read_serial_background,
            args=(
                port,
                self.BAUDRATE,
                self._stop_event,
                self._initialized_event,
                self._queue,
                self._write_queue,
            ),
            daemon=True,
        )
        self._reader_process.start()
        self._initialized_event.wait()

    @staticmethod
    def _read_serial_background(
        port: str,
        baudrate: int,
        stop_event: multiprocessing.synchronize.Event,
        initialized_event: multiprocessing.synchronize.Event,
        queue: multiprocessing.Queue,
        write_queue: multiprocessing.Queue,
    ) -> None:
        """
        Background process that continuously reads data from the serial port
        and places it into a queue for processing.
        """
        # Open the serial connection
        try:
            serial_conn = SafeSerial.create(port=port, baudrate=baudrate)
            initialized_event.set()
        except Exception as e:
            get_logger().error(f"Error opening serial connection: {e}")
            return

        try:
            while not stop_event.is_set():
                # =====
                # READ
                if serial_conn.in_waiting > 0:
                    line = serial_conn.readline()
                    assert line, "Empty line received"

                    # Put the line into the queue without blocking
                    try:
                        queue.put(line, block=False)
                    except multiprocessing.queues.Full:
                        # Queue is full; discard the line to prevent blocking
                        pass

                # =====
                # WRITE
                try:
                    config_data = write_queue.get(block=False)
                    serial_conn.write(config_data)
                except multiprocessing.queues.Empty:
                    # No data to write
                    pass

        except Exception as e:
            get_logger().error(f"Error in reader process: {e}")
            stop_event.set()
        finally:
            if serial_conn.is_open:
                serial_conn.close()

    def update(self, **kwargs) -> None:
        """
        Updates the sensor configuration with provided keyword arguments.

        Args:
            **kwargs: Configuration parameters to update. Keys must match
                the fields of SensorConfig.
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                get_logger().warning(f"Unknown config key: {key}")

        # Send the configuration to the sensor
        try:
            self._write_queue.put(self._config.pack())
        except multiprocessing.queues.Full:
            get_logger().error("Failed to send configuration to sensor")

    def accumulate(
        self,
        num_samples: int = 1,
        *,
        average: bool = True,
    ) -> np.ndarray | list[np.ndarray]:
        """
        Accumulates histogram data from the sensor.

        Args:
            num_samples (int): Number of samples to accumulate.
            average (bool, optional): If True, returns the average of the samples.
                If False, returns a list of individual samples. Defaults to True.

        Returns:
            np.ndarray | list[np.ndarray]: The accumulated histogram data,
                either averaged or as a list of samples.
        """
        histograms = []
        for _ in range(num_samples):
            began_read = False
            self._histogram.reset(self._config.resolution)
            while not self._histogram.has_data:
                try:
                    # Retrieve the next line from the queue
                    line: bytes = self._queue.get(timeout=1)
                except multiprocessing.queues.Empty:
                    # No data received in time; continue waiting
                    continue

                try:
                    line_str = line.decode("utf-8").replace("\r", "").replace("\n", "")
                    get_logger().debug(f"Processing line: {line_str}")
                except UnicodeDecodeError:
                    get_logger().error("Error decoding data")
                    continue

                if line_str.startswith("Data Count"):
                    began_read = True
                    continue

                if began_read and line_str.startswith("Agg"):
                    tokens = [
                        token.strip() for token in line_str.split(",") if token.strip()
                    ]

                    if not self._histogram.process(tokens):
                        get_logger().error("Error processing data")
                        self._histogram.reset()
                        began_read = False
                        continue

                if self._histogram.has_data:
                    histograms.append(self._histogram.get_data())
                    self._histogram.reset()
                    break  # Move on to next sample

        if num_samples == 1:
            histograms = histograms[0] if histograms else None
        elif average:
            histograms = np.mean(histograms, axis=0)
        return histograms

    @property
    def num_bins(self) -> int:
        """
        Gets the number of CNH bins in the sensor configuration.

        Returns:
            int: Number of CNH bins.
        """
        return self._config.cnh_num_bins

    @num_bins.setter
    def num_bins(self, value: int):
        """
        Sets the number of CNH bins in the sensor configuration.

        Args:
            value (int): New number of CNH bins.
        """
        self._histogram.reset()
        self.update(cnh_num_bins=value)

    @property
    def resolution(self) -> tuple[int, int]:
        """
        Gets the aggregation grid resolution.

        Returns:
            tuple[int, int]: Number of aggregation columns and rows.
        """
        return self._config.agg_cols, self._config.agg_rows

    @resolution.setter
    def resolution(self, value: tuple[int, int]):
        """
        Sets the aggregation grid resolution.

        Args:
            value (tuple[int, int]): New number of aggregation columns and rows.
        """
        self.update(agg_cols=value[0], agg_rows=value[1])

    @property
    def is_okay(self) -> bool:
        """
        Checks if the sensor is operational.

        Returns:
            bool: True if the sensor is operational, False otherwise.
        """
        return (
            self._initialized_event.is_set()
            and self._reader_process.is_alive()
            and not self._stop_event.is_set()
        )

    def close(self) -> None:
        """
        Closes the sensor connection and stops background processes.
        """
        if not self._initialized_event.is_set():
            return

        # Signal the reader process to stop
        self._stop_event.set()
        self._reader_process.join()