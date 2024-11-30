"""TMF8828 sensor driver for SPAD sensors.

The `TMF8828 \
    <https://ams-osram.com/products/sensor-solutions/\
        direct-time-of-flight-sensors-dtof/\
            ams-tmf8828-configurable-8x8-multi-zone-time-of-flight-sensor>`_
is a 8x8 multi-zone time-of-flight sensor made by AMS. It uses a wide VCSEL and supports
custom mapping of SPAD pixels to allow for 3x3, 4x4, 3x6, and 8x8 multizone output. The
:class:`~drivers.spads.tmf8828.TMF8828Sensor` class was developed to interface with the
`TMF882X Arduino Shield \
    <https://ams-osram.com/products/boards-kits-accessories/kits/\
        ams-tmf882x-evm-eb-shield-evaluation-kit>`_.
"""

import time
from enum import Enum
from pathlib import Path

import numpy as np
import pkg_resources

from cc_hardware.drivers.safe_serial import SafeSerial
from cc_hardware.drivers.sensor import SensorData
from cc_hardware.drivers.spads.spad import SPADSensor
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.registry import register

# ================

# Configuration constants
TMF882X_BINS = 128
TMF882X_OBJ_BINS = 75
TMF882X_SKIP_FIELDS = 3  # Skip the first 3 fields
TMF882X_IDX_FIELD = TMF882X_SKIP_FIELDS - 1

# ================


# Enum for SPAD IDs
class SPADID(Enum):
    ID6 = 6
    ID7 = 7
    ID15 = 15


# Enum for ranging modes
class RangeMode(Enum):
    LONG = 0
    SHORT = 1


# ================


class TMF8828Histogram(SensorData):
    """
    A class representing histogram data collected from the TMF8828 sensor. The histogram
    data is organized into multiple channels and subcaptures to capture detailed
    measurements.

    Inherits:
        SensorData: Base class for sensor data storage and processing.

    Attributes:
        num_channels (int): Total number of channels, including the calibration channel.
        active_channels_per_subcapture (list[int]): List indicating the number of active
            channels in each subcapture.
        num_subcaptures (int): The total number of subcaptures.
        spad_id (SPADID): The SPAD ID indicating the resolution of the sensor.
    """

    def __init__(
        self,
        num_channels: int,
        active_channels_per_subcapture: list[int],
        spad_id: SPADID,
    ):
        """
        Initializes the histogram with specified channels, subcaptures, and SPAD ID.

        Args:
            num_channels (int): The total number of channels including the calibration
                channel.
            active_channels_per_subcapture (list[int]): A list indicating the active
                channels in each subcapture.
            spad_id (SPADID): The SPAD ID indicating the resolution of the sensor.
        """
        super().__init__()
        self.num_channels = num_channels
        self.active_channels_per_subcapture = active_channels_per_subcapture
        self.num_subcaptures = len(active_channels_per_subcapture)
        self.spad_id = spad_id
        self._temp_data = np.zeros(
            (self.num_subcaptures, self.num_channels, TMF882X_BINS), dtype=np.int32
        )
        total_active_channels = sum(active_channels_per_subcapture)
        self._data = np.zeros((total_active_channels, TMF882X_BINS), dtype=np.int32)
        self.current_subcapture = 0
        self._has_data = False
        self._has_bad_data = False

    def reset(self) -> None:
        """
        Resets the histogram data, clearing temporary and accumulated data arrays.
        """
        self._temp_data.fill(0)
        self._data.fill(0)
        self._has_data = False
        self._has_bad_data = False
        self.current_subcapture = 0

    def process(self, row: list[str]) -> None:
        """
        Processes a single row of histogram data. Updates the internal data arrays based
        on the channel and subcapture configuration.

        Args:
            row (list[str]): A list of strings representing a row of data received from
                the sensor.
        """
        try:
            idx = int(row[TMF882X_IDX_FIELD])
        except (IndexError, ValueError):
            get_logger().error("Invalid index received.")
            self._has_bad_data = True
            return
        try:
            data = np.array(row[TMF882X_SKIP_FIELDS:], dtype=np.int32)
        except ValueError:
            get_logger().error("Invalid data received.")
            self._has_bad_data = True
            return

        if len(data) != TMF882X_BINS:
            get_logger().error(f"Invalid data length: {len(data)}")
            self._has_bad_data = True
            return

        base_idx = idx // 10
        channel = idx % 10

        if self.current_subcapture >= self.num_subcaptures:
            # Already received all subcaptures
            self._has_bad_data = True
            return

        active_channels = self.active_channels_per_subcapture[self.current_subcapture]

        if 0 <= channel <= active_channels:
            if base_idx == 0:
                self._temp_data[self.current_subcapture, channel] += data
            elif base_idx == 1:
                self._temp_data[self.current_subcapture, channel] += data * 256
            elif base_idx == 2:
                self._temp_data[self.current_subcapture, channel] += data * 256 * 256

                if channel == active_channels:
                    self.current_subcapture += 1
                    if self.current_subcapture == self.num_subcaptures:
                        self._data = self._assemble_data()
                        self._temp_data.fill(0)
                        if self._has_bad_data:
                            self.reset()
                        else:
                            self._has_data = True
        else:
            # Ignore idx values for channels that don't have measurements
            pass

    def _assemble_data(self) -> np.ndarray:
        """
        Assembles the data from all subcaptures into a single array. Handles
        reorganization of data based on the SPAD ID, especially for ID15, which requires
        pixel mapping.

        Returns:
            np.ndarray: The assembled data array.
        """
        combined_data = []
        for subcapture_index in range(self.num_subcaptures):
            active_channels = self.active_channels_per_subcapture[subcapture_index]
            data = self._temp_data[subcapture_index, 1 : active_channels + 1, :]
            combined_data.append(data)
        combined_data = np.vstack(combined_data)

        if self.spad_id == SPADID.ID15:
            # Rearrange the data according to the pixel mapping
            pixel_map = {
                1: 57,
                2: 61,
                3: 41,
                4: 45,
                5: 25,
                6: 29,
                7: 9,
                8: 13,
                11: 58,
                12: 62,
                13: 42,
                14: 46,
                15: 26,
                16: 30,
                17: 10,
                18: 14,
                21: 59,
                22: 63,
                23: 43,
                24: 47,
                25: 27,
                26: 31,
                27: 11,
                28: 15,
                31: 60,
                32: 64,
                33: 44,
                34: 48,
                35: 28,
                36: 32,
                37: 12,
                38: 16,
                41: 49,
                42: 53,
                43: 33,
                44: 37,
                45: 17,
                46: 21,
                47: 1,
                48: 5,
                51: 50,
                52: 54,
                53: 34,
                54: 38,
                55: 18,
                56: 22,
                57: 2,
                58: 6,
                61: 51,
                62: 55,
                63: 35,
                64: 39,
                65: 19,
                66: 23,
                67: 3,
                68: 7,
                71: 52,
                72: 56,
                73: 36,
                74: 40,
                75: 20,
                76: 24,
                77: 4,
                78: 8,
            }
            # Create a 3D array to hold the spatial data
            spatial_data = np.zeros((8, 8, TMF882X_BINS), dtype=combined_data.dtype)

            for idx, pixel in enumerate(pixel_map.keys()):
                # Map histogram index to pixel position in 8x8 grid
                row = (pixel_map[pixel] - 1) // 8
                col = (pixel_map[pixel] - 1) % 8
                spatial_data[row, col, :] = combined_data[idx, :]

            # Flatten the spatial data back to (64, TMF882X_BINS) if needed
            rearranged_data = spatial_data.reshape(64, TMF882X_BINS)
            return np.copy(rearranged_data)
        else:
            return np.copy(combined_data)

    def get_data(self) -> np.ndarray:
        """
        Returns a copy of the accumulated histogram data and resets the internal state.

        Returns:
            np.ndarray: The accumulated histogram data.
        """
        data = np.copy(self._data)
        self.reset()
        return data

    @property
    def has_data(self) -> bool:
        """
        Checks if the histogram has complete data for all subcaptures.

        Returns:
            bool: True if all subcaptures have been processed, False otherwise.
        """
        return self._has_data


class TMF8828Object(SensorData):
    """
    A class representing object data collected from the TMF8828 sensor. The object data
    contains information about detected objects, organized into bins.

    Inherits:
        SensorData: Base class for sensor data storage and processing.
    """

    def __init__(self):
        """
        Initializes the object data with the specified number of bins.
        """
        super().__init__()
        self._data = np.zeros(TMF882X_OBJ_BINS, dtype=np.int32)

    def reset(self) -> None:
        """
        Resets the object data, clearing the data array.
        """
        self._data.fill(0)

    def process(self, row: list[str]) -> None:
        """
        Processes a single row of object data. Updates the internal data array based on
        the received row.

        Args:
            row (list[str]): A list of strings representing a row of data received from
                the sensor.
        """
        try:
            self._data = np.array(row)[TMF882X_SKIP_FIELDS:].astype(np.int32)
        except ValueError:
            get_logger().error("Invalid data received.")
            return

    def get_data(self) -> np.ndarray:
        """
        Returns a copy of the object data.

        Returns:
            np.ndarray: The object data array.
        """
        return self._data


# ================


@register
class TMF8828Sensor(SPADSensor):
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

    def __init__(
        self,
        port: str,
        *,
        spad_id: SPADID | int = SPADID.ID6,
        setup: bool = True,
        range_mode: RangeMode = RangeMode.LONG,
    ):
        """
        Initializes the TMF8828 sensor with the specified SPAD ID, port, and setup
        parameters.

        Args:
            port (str): The port to use for communication with the sensor.

        Keyword Args:
            spad_id (SPADID | int): The SPAD ID indicating the resolution of the sensor.
                Defaults to SPADID.ID6.
            setup (bool): Whether to perform a sensor setup after initialization.
                Defaults to True.
            range_mode (RangeMode): The range mode for the sensor (LONG or SHORT).
                Defaults to LONG.
        """
        self._initialized = False
        self.spad_id = spad_id if isinstance(spad_id, SPADID) else SPADID(spad_id)
        self.range_mode = range_mode
        self._num_pixels = self._get_num_pixels()
        self.num_channels = self._get_num_channels()  # Including calibration channel
        self.active_channels_per_subcapture = self._get_active_channels_per_subcapture()

        port = port
        self._arduino = SafeSerial.create(
            port=port, baudrate=self.BAUDRATE, timeout=self.TIMEOUT
        )

        self.initialize()
        if setup:
            self.setup_sensor()

        self._histogram = TMF8828Histogram(
            self.num_channels, self.active_channels_per_subcapture, self.spad_id
        )
        self._object = TMF8828Object()

        self._initialized = True

    def _get_num_pixels(self) -> int:
        """
        Returns the number of pixels based on the SPAD ID.

        Returns:
            int: The number of pixels corresponding to the sensor's SPAD ID.
        """
        if self.spad_id == SPADID.ID6:
            return 9
        elif self.spad_id == SPADID.ID7:
            return 16
        elif self.spad_id == SPADID.ID15:
            return 64
        else:
            raise ValueError(f"Unsupported SPAD ID: {self.spad_id}")

    def _get_num_channels(self) -> int:
        """
        Returns the number of channels, including the calibration channel.

        Returns:
            int: The number of channels available in the sensor.
        """
        return 10

    def _get_active_channels_per_subcapture(self) -> list[int]:
        """
        Returns the number of active channels per subcapture based on the SPAD ID.

        Returns:
            list[int]: A list representing the number of active channels in each
                subcapture.
        """
        if self.spad_id == SPADID.ID6:
            return [9]
        elif self.spad_id == SPADID.ID7:
            return [8, 8]
        elif self.spad_id == SPADID.ID15:
            return [8, 8, 8, 8, 8, 8, 8, 8]
        else:
            raise ValueError(f"Unsupported SPAD ID: {self.spad_id}")

    def initialize(self):
        """
        Initializes the sensor by sending the initialization command.
        """
        get_logger().info("Initializing sensor...")
        self._arduino.write_and_wait_for_start_and_stop_talk("h")
        get_logger().info("Sensor initialized")

    def setup_sensor(self) -> None:
        """
        Sets up the sensor based on its SPAD ID and range mode.
        """
        get_logger().info("Setting up sensor...")

        # Reset the sensor
        self._arduino.write_and_wait_for_start_and_stop_talk("d")

        if self.spad_id in [SPADID.ID6, SPADID.ID7]:  # 3x3, 4x4
            self._arduino.write_and_wait_for_start_and_stop_talk("o")
            self._arduino.write_and_wait_for_start_and_stop_talk("E")
            if self.spad_id == SPADID.ID7:  # 4x4
                self._arduino.write_and_wait_for_start_and_stop_talk("c")
        elif self.spad_id == SPADID.ID15:  # 8x8
            self._arduino.write_and_wait_for_start_and_stop_talk("e")
        else:
            raise ValueError(f"Unsupported mode: {self.spad_id}")

        if self.range_mode == RangeMode.SHORT:
            # Default is LONG
            self._arduino.write_and_wait_for_start_and_stop_talk("O")

        self._arduino.write_and_wait_for_stop_talk("z")

        get_logger().info("Sensor setup complete")

    def read(self) -> bytes:
        """
        Reads a line of data from the sensor.

        Returns:
            bytes: The data read from the sensor.
        """
        read_line = self._arduino.readline()
        if len(read_line) > 0:
            if read_line[0] != b"#"[0]:
                get_logger().info(read_line)

        return read_line

    def write(self, data: str) -> None:
        """
        Writes data to the sensor.

        Args:
            data (str): The data to write to the sensor.
        """
        get_logger().debug(f"Writing {data}...")
        self._arduino.write(data)
        time.sleep(0.05)

    def close(self) -> None:
        """
        Closes the sensor connection.
        """
        if not self._initialized:
            return

        try:
            self._arduino.close()
        except Exception as e:
            get_logger().error(f"Error closing SafeSerial: {e}")

    def accumulate(
        self,
        num_samples: int,
        *,
        average: bool = True,
    ) -> np.ndarray | list[np.ndarray]:
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
        # Reset the serial buffer
        self._arduino.write_and_wait_for_stop_talk("s")
        self._arduino.write_and_wait_for_start_talk("m")

        histograms = []
        for _ in range(num_samples):
            get_logger().info(f"Sample {len(histograms) + 1}/{num_samples}")

            while not self._histogram.has_data:
                data = self.read()
                get_logger().debug(data)
                try:
                    data = data.decode("utf-8").replace("\r", "").replace("\n", "")
                except UnicodeDecodeError:
                    get_logger().error("Error decoding data")
                    continue
                row = data.split(",")

                if len(row) > 0 and row[0] != "":
                    if row[0] == "#Obj":
                        self._object.process(row)
                    elif row[0] == "#Raw":
                        self._histogram.process(row)

            histograms.append(self._histogram.get_data())

        # Stop the histogram reading to save USB bandwidth
        self._arduino.write_and_wait_for_stop_talk("s")

        if num_samples == 1:
            histograms = histograms[0] if histograms else None
        elif average:
            histograms = np.mean(histograms, axis=0)

        return histograms

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
            self._arduino.write_and_wait_for_start_and_stop_talk("f")
            _, calibration_data_i = self._arduino.write_and_wait_for_stop_talk(
                "l", return_data=True
            )
            self._arduino.write_and_wait_for_start_and_stop_talk("c")
            calibration_data.append(extract_calibration(calibration_data_i))
        get_logger().info("Calibration complete")

        return calibration_data

    @property
    def is_okay(self) -> bool:
        """
        Checks if the sensor is operating correctly.

        Returns:
            bool: Always returns True indicating the sensor is okay.
        """
        return True

    @property
    def num_bins(self) -> int:
        """
        Returns the number of bins in the sensor's histogram.

        Returns:
            int: The number of bins in the histogram.
        """
        return TMF882X_BINS

    @property
    def resolution(self) -> tuple[int, int]:
        """
        Returns the resolution of the sensor as a tuple (width, height).

        Returns:
            tuple[int, int]: The resolution (width, height) based on the SPAD ID.
        """
        if self.spad_id == SPADID.ID6:
            return 3, 3
        elif self.spad_id == SPADID.ID7:
            return 4, 4
        elif self.spad_id == SPADID.ID15:
            return 8, 8
        else:
            raise ValueError(f"Unsupported SPAD ID: {self.spad_id}")

    @property
    def num_pixels(self) -> int:
        """
        Returns the number of pixels based on the SPAD ID.

        Returns:
            int: The number of pixels in the sensor's active area.
        """
        return self._num_pixels
