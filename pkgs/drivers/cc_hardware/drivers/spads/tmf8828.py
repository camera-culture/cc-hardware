import re
from enum import Enum
from pathlib import Path
import time

import numpy as np
import pkg_resources

# Assume these modules are available in your project
from cc_hardware.drivers.arduino import Arduino
from cc_hardware.drivers.sensor import SensorData
from cc_hardware.drivers.spad import SPADSensor
from cc_hardware.utils.constants import C
from cc_hardware.utils.logger import get_logger

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
    def __init__(
        self,
        num_channels: int,
        active_channels_per_subcapture: list[int],
        spad_id: SPADID,
    ):
        super().__init__()
        self.num_channels = num_channels  # Including calibration channel
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
        self._temp_data.fill(0)
        self._data.fill(0)
        self._has_data = False
        self._has_bad_data = False
        self.current_subcapture = 0

    def process(self, row: list[str]) -> None:
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
        channel = idx % 10  # idx ranges from 0 to 29, channels 0-9

        if self.current_subcapture >= self.num_subcaptures:
            # Already received all subcaptures
            self._has_bad_data = True
            return

        active_channels = self.active_channels_per_subcapture[self.current_subcapture]

        # Only process valid channels
        if 0 <= channel <= active_channels:
            if base_idx == 0:
                self._temp_data[self.current_subcapture, channel] += data
            elif base_idx == 1:
                self._temp_data[self.current_subcapture, channel] += data * 256
            elif base_idx == 2:
                self._temp_data[self.current_subcapture, channel] += data * 256 * 256

                # If this is the last channel and MSB, check for more subcaptures
                if channel == active_channels:
                    self.current_subcapture += 1
                    if self.current_subcapture == self.num_subcaptures:
                        # All subcaptures received, combine data
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
        combined_data = []
        for subcapture_index in range(self.num_subcaptures):
            active_channels = self.active_channels_per_subcapture[subcapture_index]
            # Exclude calibration channel (channel 0) and limit to active channels
            data = self._temp_data[subcapture_index, 1 : active_channels + 1, :]
            combined_data.append(data)
        combined_data = np.vstack(combined_data)

        # TODO
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

        return np.copy(combined_data)

    def get_data(self) -> np.ndarray:
        data = np.copy(self._data)
        self.reset()
        return data

    @property
    def has_data(self) -> bool:
        return self._has_data


class TMF8828Object(SensorData):
    def __init__(self):
        super().__init__()
        self._data = np.zeros(TMF882X_OBJ_BINS, dtype=np.int32)

    def reset(self) -> None:
        self._data.fill(0)

    def process(self, row: list[str]) -> None:
        try:
            self._data = np.array(row)[TMF882X_SKIP_FIELDS:].astype(np.int32)
        except ValueError:
            get_logger().error("Invalid data received.")
            return

    def get_data(self) -> np.ndarray:
        return self._data


# ================


class TMF8828Sensor(SPADSensor):
    PORT: str = "/usr/local/dev/arduino-tmf8828"
    SCRIPT: Path = Path(
        pkg_resources.resource_filename(
            "cc_hardware.drivers", str(Path("data") / "tmf8828" / "tmf8828.ino")
        )
    )
    BAUDRATE: int = 2_000_000
    TIMEOUT: float = 1.0

    def __init__(
        self,
        *,
        spad_id: SPADID | int = SPADID.ID6,
        port: str | None = None,
        setup: bool = True,
        range_mode: RangeMode = RangeMode.LONG,
    ):
        self._initialized = False
        self.spad_id = spad_id if isinstance(spad_id, SPADID) else SPADID(spad_id)
        self.range_mode = range_mode
        self._num_pixels = self._get_num_pixels()
        self.num_channels = self._get_num_channels()  # Including calibration channel
        self.active_channels_per_subcapture = self._get_active_channels_per_subcapture()

        port = port or self.PORT
        self._arduino = Arduino.create(
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
        if self.spad_id == SPADID.ID6:
            return 9
        elif self.spad_id == SPADID.ID7:
            return 16
        elif self.spad_id == SPADID.ID15:
            return 64
        else:
            raise ValueError(f"Unsupported SPAD ID: {self.spad_id}")

    def _get_num_channels(self) -> int:
        # Channels 0-9, including calibration channel
        return 10

    def _get_active_channels_per_subcapture(self) -> list[int]:
        if self.spad_id == SPADID.ID6:
            return [9]
        elif self.spad_id == SPADID.ID7:
            return [8, 8]
        elif self.spad_id == SPADID.ID15:
            return [8, 8, 8, 8, 8, 8, 8, 8]
        else:
            raise ValueError(f"Unsupported SPAD ID: {self.spad_id}")

    def initialize(self):
        get_logger().info("Initializing sensor...")

        self.write_and_wait_for_start_and_stop_talk("h")

        get_logger().info("Sensor initialized")

    def setup_sensor(self) -> None:
        get_logger().info("Setting up sensor...")

        # Reset the sensor
        self.write_and_wait_for_start_and_stop_talk("d")

        if self.spad_id in [SPADID.ID6, SPADID.ID7]:  # 3x3, 4x4
            self.write_and_wait_for_start_and_stop_talk("o")
            self.write_and_wait_for_start_and_stop_talk("E")
            if self.spad_id == SPADID.ID7:  # 4x4
                self.write_and_wait_for_start_and_stop_talk("c")
        elif self.spad_id == SPADID.ID15:  # 8x8
            self.write_and_wait_for_start_and_stop_talk("e")
        else:
            raise ValueError(f"Unsupported mode: {self.spad_id}")

        if self.range_mode == RangeMode.SHORT:
            # Default is LONG
            self.write_and_wait_for_start_and_stop_talk("O")

        self.write_and_wait_for_stop_talk("z")

        get_logger().info("Sensor setup complete")

    def read(self) -> bytes:
        read_line = self._arduino.readline()
        if len(read_line) > 0:
            if read_line[0] != b"#"[0]:
                get_logger().info(read_line)

        return read_line

    def write(self, data: str) -> None:
        get_logger().debug(f"Writing {data}...")
        self._arduino.write(data)
        time.sleep(0.05)

    def wait_for_start_talk(self, timeout: float = None) -> bytes | None:
        """Wait until Arduino starts talking. Returns data if successful, None if timeout."""
        data = b""
        start_time = time.time()
        while len(data) == 0:
            if timeout is not None and time.time() - start_time > timeout:
                return None  # Timeout occurred
            data = self.read()
        return data

    def write_and_wait_for_start_talk(
        self, data: str, timeout: float | None = None, tries: int = 10
    ) -> bool:
        """Write data to Arduino and wait for it to start talking with timeout.
        If timeout happens before something is received, resend data.
        Returns True if successful, False otherwise."""
        if timeout is None:
            timeout = self.TIMEOUT

        for attempt in range(tries):
            get_logger().debug(
                f"Attempt {attempt + 1}/{tries}: Writing '{data}' and waiting for Arduino to start talking."
            )
            self.write(data)
            received_data = self.wait_for_start_talk(timeout)
            if received_data is not None:
                get_logger().debug("Arduino started talking.")
                return True
            else:
                get_logger().warning(
                    "Timeout occurred waiting for Arduino to start talking. Retrying..."
                )
                continue  # Retry
        get_logger().error(f"Failed after {tries} attempts.")
        return False

    def wait_for_stop_talk(self, timeout: float = None) -> bytes | None:
        """Wait until Arduino stops talking. Returns True if stopped before timeout, False otherwise."""
        data = b"0"
        accumulated_data = b""
        start_time = time.time()
        while len(data) > 0:
            if timeout is not None and time.time() - start_time > timeout:
                return None  # Timeout occurred
            data = self.read()
            try:
                data_str = re.sub(r"[\r\n]", "", data.decode("utf-8").strip())
                get_logger().debug(data_str)
            except UnicodeDecodeError:
                get_logger().debug(data)

            accumulated_data += data
        return accumulated_data  # Stopped talking before timeout

    def write_and_wait_for_stop_talk(
        self,
        data: str,
        timeout: float | None = None,
        tries: int = 10,
        return_data: bool = False,
    ) -> bool | tuple[bool, bytes | None]:
        """Write data to Arduino and wait for it to stop talking with timeout.
        If timeout happens before something is received, resend data.
        Returns True if successful, False otherwise."""
        if timeout is None:
            timeout = self.TIMEOUT

        for attempt in range(tries):
            get_logger().debug(
                f"Attempt {attempt + 1}/{tries}: Writing '{data}' and waiting for Arduino to start talking."
            )
            self.write(data)
            received_data = self.wait_for_start_talk(timeout)
            if received_data is None:
                get_logger().warning(
                    f"Timeout occurred waiting for Arduino to start talking. Retrying..."
                )
                continue  # Retry
            get_logger().debug("Arduino started talking. Waiting for it to stop.")
            data = self.wait_for_stop_talk(timeout)
            if data is not None:
                received_data += data
            if received_data is not None:
                get_logger().debug("Arduino stopped talking.")
                return True if not return_data else (True, received_data)
            else:
                get_logger().warning(
                    f"Timeout occurred waiting for Arduino to stop talking. Retrying..."
                )
        get_logger().error(f"Failed after {tries} attempts.")
        return False if not return_data else (False, None)

    def write_and_wait_for_start_and_stop_talk(
        self, data: str, timeout: float | None = None, tries: int = 10
    ) -> bool:
        """Write data to Arduino and wait for it to start and stop talking with timeout.
        If timeout happens before either event, resend data.
        Returns True if successful, False otherwise."""
        if timeout is None:
            timeout = self.TIMEOUT

        for attempt in range(tries):
            get_logger().debug(
                f"Attempt {attempt + 1}/{tries}: Writing '{data}' and waiting for Arduino to start and stop talking."
            )
            self.write(data)
            # Wait for Arduino to start talking
            received_data = self.wait_for_start_talk(timeout)
            if received_data is None:
                get_logger().warning(
                    "Timeout occurred waiting for Arduino to start talking. Retrying..."
                )
                continue  # Retry
            get_logger().debug("Arduino started talking. Waiting for it to stop.")
            # Wait for Arduino to stop talking
            received_data = self.wait_for_stop_talk(timeout)
            if received_data is not None:
                get_logger().debug("Arduino stopped talking.")
                return True
            else:
                get_logger().warning(
                    "Timeout occurred waiting for Arduino to stop talking. Retrying..."
                )
        get_logger().error(f"Failed after {tries} attempts.")
        return False

    def close(self) -> None:
        if not self._initialized:
            return

        try:
            self._arduino.close()
        except Exception as e:
            get_logger().error(f"Error closing Arduino: {e}")

    def accumulate(
        self,
        num_samples: int,
        *,
        average: bool = True,
    ) -> np.ndarray | list[np.ndarray]:
        # Reset the serial buffer
        self.write_and_wait_for_stop_talk("s")
        self.write_and_wait_for_start_talk("m")

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

        # Stop the histogram reading to save usb bandwidth
        self.write_and_wait_for_stop_talk("s")

        if num_samples == 1:
            histograms = histograms[0] if histograms else None
        elif average:
            histograms = np.mean(histograms, axis=0)

        return histograms

    def calibrate(self) -> list[str]:
        """This performs calibration consistent with the readme defined
        [here](https://github.com/ams-OSRAM/tmf8820_21_28_driver_arduino?tab=readme-ov-file#factory-calibration-generation-and-storing-it-for-arduino-uno).

        This completes calibration for both the tmf8828 and tmf882x modes, as well
        as both accuracy modes.
        """

        def extract_calibration(byte_data: bytes, trim_length: int = 22) -> str:
            # Convert the bytes to string
            input_string = byte_data.decode("utf-8")

            # Remove the last 'trim_length' characters and return the resulting string
            return input_string[:-trim_length].strip()

        get_logger().info("Starting calibration...")
        self.write_and_wait_for_start_and_stop_talk("f")
        _, calibration_data0 = self.write_and_wait_for_stop_talk("l", return_data=True)
        self.write_and_wait_for_start_and_stop_talk("c")
        self.write_and_wait_for_start_and_stop_talk("f")
        _, calibration_data1 = self.write_and_wait_for_stop_talk("l", return_data=True)
        get_logger().info("Calibration complete")

        return [
            extract_calibration(calibration_data0),
            extract_calibration(calibration_data1),
        ]

    @property
    def is_okay(self) -> bool:
        return True

    @property
    def num_bins(self) -> int:
        return TMF882X_BINS

    @property
    def bin_width(self) -> float:
        # Bin width is 10m / 128 bins
        return 10 / TMF882X_BINS / C

    @property
    def resolution(self) -> tuple[int, int]:
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
        return self._num_pixels
