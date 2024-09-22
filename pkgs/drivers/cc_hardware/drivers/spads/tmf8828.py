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
TMF882X_CHANNEL_MASK = np.array([0] + [1] * 9, dtype=bool)  # Use channels 1 through 10

# ================


class TMF8828Histogram(SensorData):
    def __init__(self):
        super().__init__()
        self._temp_data = np.zeros(
            (len(TMF882X_CHANNEL_MASK), TMF882X_BINS), dtype=np.int32
        )
        self._data = np.zeros((len(TMF882X_CHANNEL_MASK), TMF882X_BINS), dtype=np.int32)

    def reset(self) -> None:
        self._temp_data.fill(0)
        self._data.fill(0)
        self._has_data = False

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
            self._temp_data[idx] += data
        elif 10 <= idx <= 19:
            idx -= 10
            self._temp_data[idx] += data * 256
        elif 20 <= idx <= 29:
            idx -= 20
            self._temp_data[idx] += data * 256 * 256

            # If this is the last channel, copy the data
            if idx == 9:
                self._data = np.copy(self._temp_data)
                self._temp_data.fill(0)
                self._has_data = True

    def get_data(self) -> np.ndarray:
        data = np.copy(self._data[TMF882X_CHANNEL_MASK])
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
    ):
        self._initialized = False

        port = port or self.PORT
        self._arduino = Arduino.create(
            port=port, baudrate=self.BAUDRATE, timeout=self.TIMEOUT
        )

        self.initialize()
        if setup:
            self.setup_sensor()

        self._histogram = TMF8828Histogram()
        self._object = TMF8828Object()

        self._initialized = True

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

        self.write("e")
        self.wait_for_start_talk()
        self.wait_for_stop_talk()

        self.write("o")
        self.wait_for_start_talk()
        self.wait_for_stop_talk()

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
        return 3, 3
