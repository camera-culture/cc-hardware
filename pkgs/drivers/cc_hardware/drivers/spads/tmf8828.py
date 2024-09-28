import re
from pathlib import Path

import numpy as np
import pkg_resources

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


class TMF8828Histogram(SensorData):
    def __init__(self, num_channels: int, num_subcaptures: int, active_channels: int):
        super().__init__()
        self.num_channels = num_channels
        self.num_subcaptures = num_subcaptures
        self.active_channels = active_channels
        self._temp_data = np.zeros(
            (num_subcaptures, num_channels, TMF882X_BINS), dtype=np.int32
        )
        self._data = np.zeros(
            (self.active_channels * num_subcaptures, TMF882X_BINS), dtype=np.int32
        )
        self.current_subcapture = 0

    def reset(self) -> None:
        self._temp_data.fill(0)
        self._data.fill(0)
        self._has_data = False
        self.current_subcapture = 0

    def process(self, row: list[str]) -> None:
        idx = int(row[TMF882X_IDX_FIELD])
        try:
            data = np.array(row[TMF882X_SKIP_FIELDS:], dtype=np.int32)
        except ValueError:
            get_logger().error("Invalid data received.")
            return

        if len(data) != TMF882X_BINS:
            get_logger().error(f"Invalid data length: {len(data)}")
            return

        if 0 <= idx <= 9:
            channel = idx
            self._temp_data[self.current_subcapture, channel] += data
        elif 10 <= idx <= 19:
            channel = idx - 10
            self._temp_data[self.current_subcapture, channel] += data * 256
        elif 20 <= idx <= 29:
            channel = idx - 20
            self._temp_data[self.current_subcapture, channel] += data * 256 * 256

            # If this is the last index, check if we need more subcaptures
            if channel == 9:
                self.current_subcapture += 1
                if self.current_subcapture == self.num_subcaptures:
                    # All subcaptures received, combine data
                    self._data = self._assemble_data()
                    self._temp_data.fill(0)
                    self._has_data = True

    def _assemble_data(self) -> np.ndarray:
        # Exclude calibration channel (channel 0) and limit to active channels
        data_without_calibration = self._temp_data[
            :, 1 : self.active_channels + 1, :
        ]  # Exclude channel 0
        # Flatten the subcaptures into one array
        combined_data = data_without_calibration.reshape(-1, TMF882X_BINS)
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
    TIMEOUT: float = 0.1

    def __init__(
        self,
        *,
        port: str | None = None,
        setup: bool = True,
        mode: str = "3x3",
    ):
        self._initialized = False
        self.mode = mode
        self._num_subcaptures = self._get_num_subcaptures()
        self._num_pixels = self._get_num_pixels()
        self.num_channels = self._get_num_channels()  # Including calibration channel
        self.active_channels = (
            self._get_active_channels_per_subcapture()
        )  # Excluding calibration channel

        port = port or self.PORT
        self._arduino = Arduino.create(
            port=port, baudrate=self.BAUDRATE, timeout=self.TIMEOUT
        )

        self.initialize()
        if setup:
            self.setup_sensor()

        self._histogram = TMF8828Histogram(
            self.num_channels, self._num_subcaptures, self.active_channels
        )
        self._object = TMF8828Object()

        self._initialized = True

    def _get_num_subcaptures(self) -> int:
        if self.mode == "3x3":
            return 1
        elif self.mode == "4x4":
            return 2
        elif self.mode == "8x8":
            return 8
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _get_num_pixels(self) -> int:
        if self.mode == "3x3":
            return 9
        elif self.mode == "4x4":
            return 16
        elif self.mode == "8x8":
            return 64
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _get_num_channels(self) -> int:
        # Channels 0-9, including calibration channel
        return 10

    def _get_active_channels_per_subcapture(self) -> int:
        if self.mode == "3x3":
            return 9  # Channels 1-9
        elif self.mode == "4x4":
            return 8  # Channels 1-8
        elif self.mode == "8x8":
            return 8  # Channels 1-8
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def initialize(self):
        get_logger().info("Initializing sensor...")

        self.write("h")
        self.wait_for_start_talk()
        self.wait_for_stop_talk()

        get_logger().info("Sensor initialized")

    def setup_sensor(self) -> None:
        get_logger().info("Setting up sensor...")

        self.write("d")
        self.wait_for_stop_talk()

        if self.mode == "3x3" or self.mode == "4x4":
            if self.mode == "4x4":
                self.write("c")
                self.wait_for_start_talk()
                self.wait_for_stop_talk()
            self.write("o")
            self.wait_for_start_talk()
            self.wait_for_stop_talk()
            self.write("E")
            self.wait_for_start_talk()
            self.wait_for_stop_talk()
        elif self.mode == "8x8":
            self.write("e")
            self.wait_for_start_talk()
            self.wait_for_stop_talk()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        self.write("z")
        self.wait_for_stop_talk()

        self.write("m")

        get_logger().info("Sensor setup complete")

    def read(self) -> bytes:
        return self._arduino.readline()

    def write(self, data: str) -> None:
        self._arduino.write(data)

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
        if self.mode == "3x3":
            return 3, 3
        elif self.mode == "4x4":
            return 4, 4
        elif self.mode == "8x8":
            return 8, 8
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    @property
    def num_pixels(self) -> int:
        return self._num_pixels


# Subclasses for specific modes
class TMF8828_3x3(TMF8828Sensor):
    def __init__(self, *, port: str | None = None, setup: bool = True):
        super().__init__(port=port, setup=setup, mode="3x3")


class TMF8828_4x4(TMF8828Sensor):
    def __init__(self, *, port: str | None = None, setup: bool = True):
        super().__init__(port=port, setup=setup, mode="4x4")


class TMF8828_8x8(TMF8828Sensor):
    def __init__(self, *, port: str | None = None, setup: bool = True):
        super().__init__(port=port, setup=setup, mode="8x8")


# Factory class
class TMF8828Factory:
    @staticmethod
    def create(
        resolution: tuple[int, int],
        **kwargs,
    ) -> TMF8828Sensor:
        if resolution == (3, 3):
            return TMF8828_3x3(**kwargs)
        elif resolution == (4, 4):
            return TMF8828_4x4(**kwargs)
        elif resolution == (8, 8):
            return TMF8828_8x8(**kwargs)
        else:
            raise ValueError(f"Unsupported resolution: {resolution}")
