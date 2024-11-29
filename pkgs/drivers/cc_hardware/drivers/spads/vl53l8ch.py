import multiprocessing
from pathlib import Path

import numpy as np
import pkg_resources

from cc_hardware.drivers.safe_serial import SafeSerial
from cc_hardware.drivers.sensor import SensorData
from cc_hardware.drivers.spads.spad import SPADSensor
from cc_hardware.utils.constants import C
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.registry import register

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
                get_logger().error(
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
            self._pixel_histograms.append(np.clip(bin_vals, 0, None))
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


@register
class VL53L8CHSensor(SPADSensor):
    SCRIPT: Path = Path(
        pkg_resources.resource_filename(
            "cc_hardware.drivers", str(Path("data") / "vl53l8ch" / "build" / "makefile")
        )
    )

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

        # Use multiprocessing Queue for inter-process communication
        self._queue = multiprocessing.Queue(maxsize=100)
        self._stop_event = multiprocessing.Event()

        # Start the reader process
        self._reader_process = multiprocessing.Process(
            target=self._read_serial_background,
            args=(port, debug, self._queue, self._stop_event),
            daemon=True,
        )
        self._reader_process.start()

        self._histogram = VL53L8CHHistogram()
        self._num_bins = 24
        self.change_num_bins(self._num_bins)

        self._initialized = True

    def _get_num_pixels(self) -> int:
        return 16

    def write(self, data: str) -> None:
        raise NotImplementedError
        # get_logger().debug(f"Writing {data}...")
        # padded_str = data.rjust(10, " ")
        # self._serial.write(padded_str)
        # time.sleep(0.05)

    def change_num_bins(self, num_bins: int) -> None:
        # Implement a way to send this command to the reader process if necessary
        self._num_bins = num_bins

    def close(self) -> None:
        if not self._initialized:
            return

        # Signal the reader process to stop
        self._stop_event.set()
        self._reader_process.join()

    @staticmethod
    def _read_serial_background(port, debug, queue, stop_event):
        # Initialize serial connection in this process
        serial_conn = SafeSerial.create(
            lock_type="multiprocessing",
            port=port,
            baudrate=VL53L8CHSensor.BAUDRATE,
            timeout=VL53L8CHSensor.TIMEOUT,
            wait=1,
        )

        while not stop_event.is_set():
            line = serial_conn.readline()
            if not line:
                continue

            # Put the line into the queue without blocking
            try:
                queue.put(line, block=False)
            except multiprocessing.queues.Full:
                # Queue is full; discard the line to prevent blocking
                continue

        serial_conn.close()

    def accumulate(
        self,
        num_samples: int,
        *,
        average: bool = True,
    ) -> np.ndarray | list[np.ndarray]:
        histograms = []
        for _ in range(num_samples):
            began_read = False
            self._histogram.reset()
            while not self._histogram.has_data:
                try:
                    # Retrieve the next line from the queue
                    line: str = self._queue.get(timeout=1)
                except multiprocessing.queues.Empty:
                    # No data received in time; continue waiting
                    continue

                try:
                    line = line.decode("utf-8").replace("\r", "").replace("\n", "")
                    get_logger().debug(f"Processing line: {line}")
                except UnicodeDecodeError:
                    get_logger().error("Error decoding data")
                    continue

                if line.startswith("Zone"):
                    continue

                if line.startswith("Print data"):
                    began_read = True
                    continue

                if began_read:
                    if line.startswith("Agg"):
                        tokens = [
                            token.strip() for token in line.split(",") if token.strip()
                        ]
                        if not self._histogram.process(tokens):
                            get_logger().error("Error processing data")
                            self._histogram.reset()
                            began_read = False
                            continue
                    if self._histogram.has_data:
                        histograms.append(self._histogram.get_data())
                        self._histogram.reset()
                        began_read = False
                        break  # Move on to next sample

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
