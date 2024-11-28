import numpy as np

from cc_hardware.drivers.safe_serial import SafeSerial
from cc_hardware.drivers.sensor import SensorData
from cc_hardware.drivers.spads.spad import SPADSensor
from cc_hardware.utils.constants import C
from cc_hardware.utils.logger import get_logger

# ================


class VL53L8CHHistogram(SensorData):
    def __init__(self):
        super().__init__()

        self._pixel_histograms = []
        self._has_data = False

    def reset(self) -> None:
        self._has_data = False
        self._pixel_histograms = []

    def process(self, row: list[str]) -> bool:
        try:
            agg_num = int(row[1])
            if agg_num != len(self._pixel_histograms):
                print(
                    "Mismatched histogram message: "
                    f"{agg_num} != {len(self._pixel_histograms)}"
                )
                return False
        except ValueError:
            get_logger().error("Invalid data formatting received.")
            return False

        try:
            ambient = float(row[3])
            bin_vals = [float(val) - ambient for val in row[5:]]
            self._pixel_histograms.append(np.array(bin_vals))
        except ValueError:
            get_logger().error("Invalid data formatting received.")
            return False

        if len(self._pixel_histograms) == 16:
            self._data = np.array(self._pixel_histograms)
            self._has_data = True

        return True

    def get_data(self) -> np.ndarray:
        data = np.copy(self._data)
        self.reset()
        return data

    @property
    def has_data(self) -> bool:
        return self._has_data


# ================


class VL53L8CHSensor(SPADSensor):
    BAUDRATE: int = 921_600
    TIMEOUT: float = 1.0

    def __init__(
        self,
        *,
        port: str | None = None,
        debug: bool = False,
    ):
        self._initialized = False
        self._num_pixels = self._get_num_pixels()

        self._serial = SafeSerial.create(
            port=port, baudrate=self.BAUDRATE, timeout=self.TIMEOUT, wait=1
        )

        self._debug = debug
        self._histogram = VL53L8CHHistogram()

        self._num_bins = 24
        self.change_num_bins(self._num_bins)

        self._initialized = True

    def _get_num_pixels(self) -> int:
        return 16

    def read(self) -> bytes:
        read_line = self._serial.readline()
        if self._debug and len(read_line) > 0:
            if read_line[0] != b"#"[0]:
                # print debug lines
                try:
                    read_line_str = (
                        read_line.decode("utf-8").replace("\r", "").replace("\n", "")
                    )
                    if len(read_line_str) > 5:
                        if read_line_str[:5] == "Debug":
                            get_logger().info(read_line_str)
                except UnicodeDecodeError:
                    get_logger().error(read_line)

        return read_line

    def write(self, data: str) -> None:
        get_logger().debug(f"Writing {data}...")
        padded_str = data.rjust(10, " ")
        self._serial.write(padded_str)

    def change_num_bins(self, num_bins: int) -> None:
        get_logger().info(f"Changing num_bins to {num_bins}")
        self.write(f"{num_bins}")
        self._histogram = VL53L8CHHistogram()
        self._num_bins = num_bins

    def close(self) -> None:
        if not self._initialized:
            return

        try:
            self._serial.close()
        except Exception as e:
            get_logger().error(f"Error closing serial: {e}")

    def accumulate(
        self,
        num_samples: int,
        *,
        average: bool = True,
    ) -> np.ndarray | list[np.ndarray]:
        histograms = []
        for _ in range(num_samples):
            began_read = False
            while not self._histogram.has_data:
                line = self.read()
                try:
                    line = line.decode("utf-8").replace("\n", "")
                    get_logger().debug(line)
                except UnicodeDecodeError:
                    get_logger().error("Error decoding data")
                    continue

                if line[:10] == "Print data":
                    began_read = True
                    continue

                if began_read:
                    if len(line) > 3 and line[:3] == "Agg":
                        tokens = []
                        for raw_token in line.split(","):
                            token = raw_token.strip()
                            if len(token) > 0:
                                tokens.append(token)

                        if not self._histogram.process(tokens):
                            get_logger().error("Error processing data")
                            self._histogram.reset()
                            began_read = False
                            continue

            histograms.append(self._histogram.get_data())

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
        return self._num_bins

    @property
    def bin_width(self) -> float:
        return 10 / self._num_bins / C

    @property
    def resolution(self) -> tuple[int, int]:
        return 4, 4

    @property
    def num_pixels(self) -> int:
        return self._num_pixels
