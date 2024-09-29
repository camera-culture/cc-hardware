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

    def reset(self) -> None:
        self._temp_data.fill(0)
        self._data.fill(0)
        self._has_data = False
        self.current_subcapture = 0

    def process(self, row: list[str]) -> None:
        try:
            idx = int(row[TMF882X_IDX_FIELD])
        except (IndexError, ValueError):
            get_logger().error("Invalid index received.")
            return
        try:
            data = np.array(row[TMF882X_SKIP_FIELDS:], dtype=np.int32)
        except ValueError:
            get_logger().error("Invalid data received.")
            return

        if len(data) != TMF882X_BINS:
            get_logger().error(f"Invalid data length: {len(data)}")
            return

        base_idx = idx // 10
        channel = idx % 10  # idx ranges from 0 to 29, channels 0-9

        if self.current_subcapture >= self.num_subcaptures:
            # Already received all subcaptures
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
        # if self.spad_id == SPADID.ID15:
        #     # Rearrange the data according to the pixel mapping
        #     pixel_mapping = [
        #         (4, 7), (5, 8), (6, 7), (7, 8), (4, 8), (5, 9), (6, 8), (7, 9),
        #         (0, 7), (1, 8), (2, 7), (3, 8), (0, 8), (1, 9), (2, 8), (3, 9),
        #         (4, 5), (5, 6), (6, 5), (7, 6), (4, 6), (5, 7), (6, 6), (7, 7),
        #         (0, 5), (1, 6), (2, 5), (3, 6), (0, 6), (1, 7), (2, 6), (3, 7),
        #         (4, 3), (5, 4), (6, 3), (7, 4), (4, 4), (5, 5), (6, 4), (7, 5),
        #         (0, 3), (1, 4), (2, 3), (3, 4), (0, 4), (1, 5), (2, 4), (3, 5),
        #         (4, 1), (5, 2), (6, 1), (7, 2), (4, 2), (5, 3), (6, 2), (7, 3),
        #         (0, 1), (1, 2), (2, 1), (3, 2), (0, 2), (1, 3), (2, 2), (3, 3),
        #     ]
        #     # Create a 3D array to hold the spatial data
        #     spatial_data = np.zeros((8, 8, TMF882X_BINS), dtype=combined_data.dtype)
        #     for idx in range(combined_data.shape[0]):
        #         row, col = pixel_mapping[idx]
        #         spatial_data[row, col - 2, :] = combined_data[idx, :]
        #     # Flatten the spatial data back to (64, TMF882X_BINS) if needed
        #     rearranged_data = spatial_data.reshape(64, TMF882X_BINS)
        #     return np.copy(rearranged_data)
        # else:
        #     return np.copy(combined_data)

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
        spad_id: SPADID | int,
        port: str | None = None,
        setup: bool = True,
    ):
        self._initialized = False
        self.spad_id = spad_id if isinstance(spad_id, SPADID) else SPADID(spad_id)
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

        self.write("h")
        self.wait_for_start_talk()
        self.wait_for_stop_talk()

        get_logger().info("Sensor initialized")

    def setup_sensor(self) -> None:
        get_logger().info("Setting up sensor...")

        # Reset the sensor
        self.write("d")
        self.wait_for_start_talk()
        self.wait_for_stop_talk()

        if self.spad_id in [SPADID.ID6, SPADID.ID7]:  # 3x3, 4x4
            self.write("o")  # Switch to TMF882x mode
            self.wait_for_start_talk()
            self.wait_for_stop_talk()
            self.write("E")
            self.wait_for_start_talk()
            self.wait_for_stop_talk()
            if self.spad_id == SPADID.ID7:  # 4x4
                self.write("c")  # Move to the next configuration
                self.wait_for_start_talk()
                self.wait_for_stop_talk()
        elif self.spad_id == SPADID.ID15:  # 8x8
            self.write("e")
            self.wait_for_start_talk()
            self.wait_for_stop_talk()
        else:
            raise ValueError(f"Unsupported mode: {self.spad_id}")

        self.write("z")
        self.wait_for_stop_talk()

        get_logger().info("Sensor setup complete")

    def read(self) -> bytes:
        read_line = self._arduino.readline()
        if len(read_line) > 0:
            if read_line[0] != b'#'[0]:
                get_logger().info(read_line)

        return read_line

    def write(self, data: str) -> None:
        get_logger().debug(f"Writing {data}...")
        self._arduino.write(data)
        time.sleep(0.05)

    def wait_for_start_talk(self) -> bytes:
        """Wait until Arduino starts talking."""
        data = b""
        while len(data) == 0:
            data = self.read()
        return data

    def wait_for_stop_talk(self) -> None:
        """Wait until Arduino stops talking."""
        data = b"0"
        while len(data) > 0:
            data = self.read()
            try:
                data_str = re.sub(r"[\r\n]", "", data.decode("utf-8").strip())
                get_logger().debug(data_str)
            except UnicodeDecodeError:
                get_logger().debug(data)

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
        self.write("s")
        self.wait_for_stop_talk()
        self.write("m")
        self.wait_for_start_talk()

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

        self.write("s")
        self.wait_for_stop_talk()

        if num_samples == 1:
            histograms = histograms[0] if histograms else None
        elif average:
            histograms = np.mean(histograms, axis=0)

        return histograms

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
