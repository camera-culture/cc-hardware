"""TMF8828 sensor driver for SPAD sensors.

The `TMF8828 \
    <https://ams-osram.com/products/sensor-solutions/\
        direct-time-of-flight-sensors-dtof/\
            ams-tmf8828-configurable-8x8-multi-zone-time-of-flight-sensor>`_
is a 8x8 multi-zone time-of-flight sensor made by AMS. It uses a wide VCSEL and supports
custom mapping of SPAD pixels to allow for 3x3, 4x4, 3x6, and 8x8 multizone output. The
:class:`~cc_hardware.drivers.spads.tmf8828.TMF8828Sensor` class was developed to interface with the
`TMF882X Arduino Shield \
    <https://ams-osram.com/products/boards-kits-accessories/kits/\
        ams-tmf882x-evm-eb-shield-evaluation-kit>`_.
"""

import multiprocessing
import multiprocessing.synchronize
import time
from enum import Enum
from pathlib import Path

import numpy as np
import pkg_resources
from tqdm import tqdm

from cc_hardware.drivers.safe_serial import SafeSerial
from cc_hardware.drivers.spads.spad import (
    SPADDataType,
    SPADSensor,
    SPADSensorConfig,
    SPADSensorData,
)
from cc_hardware.utils import config_wrapper, get_logger

# ================

# Configuration constants
TMF882X_SKIP_FIELDS = 3  # Skip the first 3 fields
TMF882X_IDX_FIELD = TMF882X_SKIP_FIELDS - 1

# ================


# Enum for SPAD IDs
class SPADID(Enum):
    ID6 = 6
    ID7 = 7
    ID15 = 15

    @property
    def num_channels(self) -> int:
        """
        Returns the number of channels based on the SPAD ID.

        Returns:
            int: The number of channels corresponding to the SPAD ID.
        """
        return 10

    @property
    def active_channels_per_subcapture(self) -> list[int]:
        """
        Returns the number of active channels per subcapture based on the SPAD ID.

        Returns:
            list[int]: A list representing the number of active channels in each
                subcapture.
        """
        if self == SPADID.ID6:
            return [9]
        elif self == SPADID.ID7:
            return [8, 8]
        elif self == SPADID.ID15:
            return [8, 8, 8, 8, 8, 8, 8, 8]
        else:
            raise ValueError(f"Unsupported SPAD ID: {self}")

    @property
    def resolution(self) -> tuple[int, int]:
        """
        Returns the resolution of the sensor based on the SPAD ID.

        Returns:
            tuple[int, int]: The resolution (width, height) corresponding to the SPAD ID.
        """
        if self == SPADID.ID6:
            return 3, 3
        elif self == SPADID.ID7:
            return 4, 4
        elif self == SPADID.ID15:
            return 8, 8
        else:
            raise ValueError(f"Unsupported SPAD ID: {self}")

    @property
    def fov(self) -> tuple[float, float]:
        """
        Returns the field of view (FOV) in degrees based on the SPAD ID.

        Returns:
            tuple[float, float]: The field of view (FOVx, FOVy) corresponding to the
                SPAD ID.
        """
        if self == SPADID.ID6:
            return 41.0, 52.0
        elif self == SPADID.ID7:
            return 41.0, 52.0
        elif self == SPADID.ID15:
            return 41.0, 52.0
        else:
            raise ValueError(f"Unsupported SPAD ID: {self}")


# Enum for ranging modes
class RangeMode(Enum):
    LONG = 0
    SHORT = 1

    @property
    def timing_resolution(self) -> float:
        """
        Returns the timing resolution for the range mode.

        Returns:
            float: The timing resolution in seconds.
        """
        if self == RangeMode.LONG:
            return 260e-12
        elif self == RangeMode.SHORT:
            return 100e-12
        else:
            raise ValueError(f"Unsupported range mode: {self}")


@config_wrapper
class TMF8828Config(SPADSensorConfig):
    """Configuration for the TMF8828 sensor.

    Attributes:
        port (str | None): The port to use for communication with the sensor.

        spad_id (SPADID): The SPAD ID indicating the resolution of the sensor.
        range_mode (RangeMode): The range mode for the sensor (LONG or SHORT).
    """

    port: str | None = None

    spad_id: SPADID = SPADID.ID6
    range_mode: RangeMode = RangeMode.LONG

    # placeholders (to be filled in by __post_init__)
    height: int = 0
    width: int = 0
    fovx: float = 0.0
    fovy: float = 0.0
    timing_resolution: float = 0.0

    num_bins: int = 128

    def __post_init__(self):
        self.height, self.width = self.spad_id.resolution
        self.timing_resolution = self.range_mode.timing_resolution
        self.fovx, self.fovy = self.spad_id.fov


# ================


class TMF8828Data(SPADSensorData[TMF8828Config]):
    """
    Processes and stores both histogram and target data from the VL53L8CH sensor.

    This class handles the accumulation and processing of histogram bins
    and per-pixel target information, keeping them aligned by pixel index.
    """

    def __init__(self, config: TMF8828Config):
        super().__init__(config)
        self.active_channels_per_subcapture = (
            config.spad_id.active_channels_per_subcapture
        )
        self.num_subcaptures = len(self.active_channels_per_subcapture)

    def reset(self) -> None:
        super().reset()
        self._has_bad_data = False
        self.current_subcapture = 0
        self._last_idx = -1

    def process(self, row: list[str]) -> bool:
        """
        Processes a row of data, routing to histogram or target handlers.

        Args:
            row (list[str]): A row of string values from sensor output.

        Returns:
            bool: True if processing succeeds, False otherwise.
        """

        return self._process_histogram(row)

    def _process_histogram(self, row: list[str]) -> bool:
        """
        Processes a histogram row for a single pixel.

        The row is in the following format:
        "Data Count, Pixel Index, ..., Distance, ..., Ambient, ..., Bins, Bin 0, Bin 1, ..."

        Args:
            row (list[str]): A histogram data row.

        Returns:
            bool: True if valid, False otherwise.
        """
        histogram: dict = self._data.setdefault(SPADDataType.HISTOGRAM, {})

        try:
            idx = int(row[TMF882X_IDX_FIELD])
        except (IndexError, ValueError):
            get_logger().error("Invalid index received.")
            self._has_bad_data = True
            return False

        if idx != self._last_idx + 1 and not self._has_bad_data:
            self._has_bad_data = True
            return False
        self._last_idx = idx

        try:
            data = np.array(row[TMF882X_SKIP_FIELDS:], dtype=np.int32)
        except ValueError:
            get_logger().error("Invalid data received.")
            self._has_bad_data = True
            return False

        if len(data) != self._config.num_bins:
            get_logger().error(f"Invalid data length: {len(data)}")
            self._has_bad_data = True
            return False

        base_idx = idx // 10
        channel = idx % 10

        if self.current_subcapture >= self.num_subcaptures:
            # Already received all subcaptures
            self._has_bad_data = True
            get_logger().error(
                f"Received data for subcapture {self.current_subcapture}, "
                f"but only {self.num_subcaptures} subcaptures expected."
            )
            return False

        active_channels = self.active_channels_per_subcapture[self.current_subcapture]

        subcapture = histogram.setdefault(
            self.current_subcapture,
            np.zeros((active_channels + 1, self._config.num_bins), dtype=np.int32),
        )
        if not (0 <= channel <= active_channels):
            return

        if base_idx == 0:
            subcapture[channel] += data
        elif base_idx == 1:
            subcapture[channel] += data * 256
        elif base_idx == 2:
            subcapture[channel] += data * 256 * 256

        if base_idx == 2 and channel == active_channels:
            self.current_subcapture += 1
            if self.current_subcapture == self.num_subcaptures:
                return self._finalize()

        return True

    def _finalize(self) -> bool:
        """
        Finalizes data when all pixels processed, storing combined output.
        """

        histogram = self._data[SPADDataType.HISTOGRAM]

        # import pdb; pdb.set_trace()
        combined_data = []
        for subcapture_index in range(self.num_subcaptures):
            active_channels = self.active_channels_per_subcapture[subcapture_index]
            data = histogram[subcapture_index][1 : active_channels + 1, :]
            combined_data.append(data)
        combined_data = np.vstack(combined_data)
        # Remove ambient data; ambient light is in first 7 bins
        # combined_data -= np.median(combined_data[:, :7], axis=1)[:, np.newaxis].astype(int)

        if self._config.spad_id == SPADID.ID15:
            # Rearrange the data according to the pixel mapping
            # fmt: off
            # flake8: noqa
            pixel_map = {
                1: 57,  2: 61,  3: 41,  4: 45,  5: 25,  6: 29,  7:  9,  8: 13,
               11: 58, 12: 62, 13: 42, 14: 46, 15: 26, 16: 30, 17: 10, 18: 14,
               21: 59, 22: 63, 23: 43, 24: 47, 25: 27, 26: 31, 27: 11, 28: 15,
               31: 60, 32: 64, 33: 44, 34: 48, 35: 28, 36: 32, 37: 12, 38: 16,
               41: 49, 42: 53, 43: 33, 44: 37, 45: 17, 46: 21, 47:  1, 48:  5,
               51: 50, 52: 54, 53: 34, 54: 38, 55: 18, 56: 22, 57:  2, 58:  6,
               61: 51, 62: 55, 63: 35, 64: 39, 65: 19, 66: 23, 67:  3, 68:  7,
               71: 52, 72: 56, 73: 36, 74: 40, 75: 20, 76: 24, 77:  4, 78:  8,
            }
            # fmt: on
            # flake8: noqa
            # Create a 3D array to hold the spatial data
            spatial_data = np.zeros(
                (8, 8, self._config.num_bins), dtype=combined_data.dtype
            )

            for idx, pixel in enumerate(pixel_map.keys()):
                # Map histogram index to pixel position in 8x8 grid
                row = (pixel_map[pixel] - 1) // 8
                col = (pixel_map[pixel] - 1) % 8
                spatial_data[row, col, :] = combined_data[idx, :]

            # Flatten the spatial data back to (64, num_bins) if needed
            rearranged_data = spatial_data.reshape(64, self._config.num_bins)
            histogram = np.copy(rearranged_data)
        else:
            histogram = np.copy(combined_data)

        self._ready_data[SPADDataType.HISTOGRAM] = histogram.reshape(
            self._config.height, self._config.width, self._config.num_bins
        )

        self._last_idx = -1
        # if self._has_bad_data:
        #     self.reset()
        #     return False
        return True


# ================


class TMF8828Sensor(SPADSensor[TMF8828Config]):
    """
    A class representing the TMF8828 sensor, a specific implementation of a SPAD sensor.
    The TMF8828 sensor collects histogram data across multiple channels and subcaptures,
    enabling high-resolution depth measurements.

    Inherits:
        SPADSensor: Base class for SPAD sensors that defines common methods and
            properties.

    Attributes:
        SCRIPT (Path): The default path to the sensor's Arduino script.
        BAUDRATE (int): The communication baud rate.
        TIMEOUT (float): The timeout value for sensor communications.
    """

    SCRIPT: Path = Path(
        pkg_resources.resource_filename(
            "cc_hardware.drivers", str(Path("data") / "tmf8828" / "tmf8828.ino")
        )
    )
    BAUDRATE: int = 2_000_000
    TIMEOUT: float = 1.0

    def __init__(self, config: TMF8828Config):
        """
        Initializes the TMF8828 sensor with the specified configuration.

        Args:
            config (TMF8828Config): The configuration for the sensor.
        """
        super().__init__(config)

        self.spad_id = config.spad_id
        self.range_mode = config.range_mode

        self._queue = multiprocessing.Queue(maxsize=self.config.num_pixels)
        self._write_queue = multiprocessing.Queue(maxsize=10)
        self._initialized_event = multiprocessing.Event()
        self._stop_event = multiprocessing.Event()

        # self._histogram = TMF8828Histogram(self.spad_id)
        self._data = TMF8828Data(config)

        # Start the reader process
        self._reader_process = multiprocessing.Process(
            target=self._read_serial_background,
            args=(
                dict(
                    port=config.port,
                    baudrate=self.BAUDRATE,
                    timeout=self.TIMEOUT,
                    one=True,
                ),
                self.spad_id,
                self.range_mode,
                self._stop_event,
                self._initialized_event,
                self._queue,
                self._write_queue,
            ),
            daemon=True,
        )
        self._reader_process.start()
        self._initialized_event.wait()

    @property
    def config(self) -> TMF8828Config:
        """
        Returns the configuration for the sensor.

        Returns:
            TMF8828Config: The configuration for the sensor.
        """
        return self._config

    @staticmethod
    def _read_serial_background(
        serial_kwargs: dict[str, str],
        spad_id: SPADID,
        range_mode: RangeMode,
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
            serial_conn = SafeSerial.create(**serial_kwargs)

            get_logger().info("Initializing sensor...")
            serial_conn.write_and_wait_for_start_and_stop_talk("h")
            get_logger().info("Sensor initialized")

            get_logger().info("Setting up sensor...")
            # Reset the sensor
            serial_conn.write_and_wait_for_start_and_stop_talk("d")

            if spad_id in [SPADID.ID6, SPADID.ID7]:  # 3x3, 4x4
                serial_conn.write_and_wait_for_start_and_stop_talk("o")
                serial_conn.write_and_wait_for_start_and_stop_talk("E")
                if spad_id == SPADID.ID7:  # 4x4
                    serial_conn.write_and_wait_for_start_and_stop_talk("c")
            elif spad_id == SPADID.ID15:  # 8x8
                serial_conn.write_and_wait_for_start_and_stop_talk("e")
            else:
                raise ValueError(f"Unsupported mode: {spad_id}")

            if range_mode == RangeMode.SHORT:
                # Default is LONG
                serial_conn.write_and_wait_for_start_and_stop_talk("O")

            serial_conn.write_and_wait_for_stop_talk("z")

            # Start measuring
            serial_conn.write_and_wait_for_start_talk("m")

            get_logger().info("Sensor setup complete")

        except Exception as e:
            get_logger().error(f"Error opening serial connection: {e}")
            stop_event.set()
            return
        finally:
            initialized_event.set()

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
                while not write_queue.empty():
                    try:
                        data, wait_for_stop, wait_for_start = write_queue.get(
                            block=False
                        )
                        if wait_for_start and wait_for_stop:
                            serial_conn.write_and_wait_for_start_and_stop_talk(data)
                        elif wait_for_start:
                            serial_conn.write_and_wait_for_start_talk(data)
                        elif wait_for_stop:
                            serial_conn.write_and_wait_for_stop_talk(data)
                        else:
                            serial_conn.write(data)
                    except multiprocessing.queues.Empty:
                        # Queue is empty; continue processing
                        pass

        except Exception as e:
            get_logger().error(f"Error in reader process: {e}")
            stop_event.set()
        finally:
            if serial_conn.is_open:
                serial_conn.close()

    def accumulate(
        self,
        num_samples: int = 1,
    ) -> list[dict[SPADDataType, np.ndarray]] | dict[SPADDataType, np.ndarray]:
        """
        Accumulates histogram samples from the sensor.

        Args:
            num_samples (int): The number of samples to accumulate.
            average (bool): Whether to average the accumulated samples. Defaults to
                True.

        Returns:
            np.ndarray | list[np.ndarray]: The accumulated histogram data, averaged if
                requested.
        """

        samples = []
        for _ in range(num_samples):
            self._data.reset()

            while not self._data.has_data:
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

                if line_str.startswith("#Raw"):
                    row = line_str.split(",")
                    self._data.process(row)

            samples.append(self._data.get_data())

        return samples[0] if num_samples == 1 else samples

    def calibrate(self, configurations: int = 2) -> list[str]:
        """
        Performs calibration on the sensor. This will run calibration for each
        configuration.

        Args:
            configurations (int): The number of configurations to calibrate. Defaults
                to 2.

        Returns:
            list[str]: A list containing the calibration strings for different modes.
        """

        def extract_calibration(byte_data: bytes, trim_length: int = 22) -> str:
            input_string = byte_data.decode("utf-8")
            return input_string[:-trim_length].strip()

        get_logger().info("Starting calibration...")
        calibration_data = []
        for i in range(configurations):
            get_logger().info(f"Calibrating configuration {i + 1}")
            self._write_queue.put(("f", True, True))
            self._write_queue.put(("l", False, True))
            try:
                calibration_data_i = self._queue.get(timeout=10)
            except multiprocessing.queues.Empty:
                get_logger().error("Calibration data not received")
                break
            self._write_queue.put(("c", True, True))
            calibration_data.append(extract_calibration(calibration_data_i))
        get_logger().info("Calibration complete")

        return calibration_data

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
        if not hasattr(self, "_initialized_event"):
            return

        if not self._initialized_event.is_set():
            return

        # Stop the histogram reading
        self._write_queue.put(("s", True, False))
        time.sleep(0.5)

        # Signal the reader process to stop
        self._stop_event.set()
        self._reader_process.join()
