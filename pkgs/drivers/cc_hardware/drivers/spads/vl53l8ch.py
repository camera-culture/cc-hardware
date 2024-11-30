import re
from enum import Enum
from pathlib import Path
import time
import threading
from typing import override

import numpy as np
import pkg_resources

# Assume these modules are available in your project
from cc_hardware.drivers.arduino import Arduino
from cc_hardware.drivers.sensor import SensorData
from cc_hardware.drivers.spad import SPADSensor
from cc_hardware.utils.constants import C
from cc_hardware.utils.blocking_deque import BlockingDeque
from cc_hardware.utils.logger import get_logger

# ================

# Configuration constants
TMF882X_BINS = 128
TMF882X_OBJ_BINS = 75
TMF882X_SKIP_FIELDS = 3  # Skip the first 3 fields
TMF882X_IDX_FIELD = TMF882X_SKIP_FIELDS - 1

# ================


# # Enum for SPAD IDs
# class SPADID(Enum):
#     ID6 = 6
#     ID7 = 7
#     ID15 = 15


# ================


class VL53L8CHHistogram(SensorData):
    def __init__(
        self,
        # num_channels: int,
        # active_channels_per_subcapture: list[int],
        # spad_id: SPADID,
    ):
        super().__init__()
        # self.num_channels = num_channels  # Including calibration channel
        # self.active_channels_per_subcapture = active_channels_per_subcapture
        # self.num_subcaptures = len(active_channels_per_subcapture)
        # self.spad_id = spad_id
        # self._temp_data = np.zeros(
        #     (self.num_subcaptures, self.num_channels, TMF882X_BINS), dtype=np.int32
        # )
        # total_active_channels = sum(active_channels_per_subcapture)
        # self._data = np.zeros((total_active_channels, TMF882X_BINS), dtype=np.int32)
        # self.current_subcapture = 0
        self._pixel_histograms = []
        self._has_data = False

    def reset(self) -> None:
        # pass
        # self._temp_data.fill(0)
        # self._data.fill(0)
        self._has_data = False
        self._pixel_histograms = []
        # self.current_subcapture = 0

    def process(self, row: list[str]) -> None:
        
        # print(f'tokens: {row}')
        try:
            agg_num = int(row[1])
            if agg_num != len(self._pixel_histograms):
                print(f'Mismatched histogram message: {agg_num} != {len(self._pixel_histograms)}')
                return  # should reset the histogram
        except ValueError:
            get_logger().error("Invalid data formatting received.")
            return

        try:
            ambient = float(row[3])
            bin_vals = [float(val) - ambient for val in row[5:]]
            # bin_vals = [float(val) for val in row[5:]]
            self._pixel_histograms.append(np.array(bin_vals))
        except ValueError:
            get_logger().error("Invalid data formatting received.")
            return
        
        # if len(self._pixel_histograms) == 16:
        #     self._data = np.array(self._pixel_histograms)
        #     self._has_data = True


    def _assemble_data(self, num_hist_expected: int) -> None:
        # if len(self._pixel_histograms) == 16:
        if len(self._pixel_histograms) == num_hist_expected:
            self._data = np.array(self._pixel_histograms)
            self._has_data = True
        else:
            self.reset()

    def get_data(self) -> np.ndarray:
        data = np.copy(self._data)
        self.reset()
        return data

    @property
    def has_data(self) -> bool:
        return self._has_data


class VL53L8CHSensor(SPADSensor):
    PORT: str = "/dev/cu.usbmodem1103"
    # SCRIPT: Path = Path(
    #     pkg_resources.resource_filename(
    #         "cc_hardware.drivers", str(Path("data") / "tmf8828" / "tmf8828.ino")
    #     )
    # )
    BAUDRATE: int = 921_600
    TIMEOUT: float = 1.0

    def __init__(
        self,
        *,
        # spad_id: SPADID | int,
        port: str | None = None,
        # setup: bool = True,
        init: bool = True,  # If you want multiple references to the same sensor for I/O operations
        # only initialize the one reading data
        debug: bool = False,
    ):
        self._initialized = False
        # self.spad_id = spad_id if isinstance(spad_id, SPADID) else SPADID(spad_id)
        # self.num_channels = self._get_num_channels()  # Including calibration channel
        # self.active_channels_per_subcapture = self._get_active_channels_per_subcapture()

        port = port or self.PORT
        self._arduino = Arduino.create(
            port=port, baudrate=self.BAUDRATE, timeout=self.TIMEOUT,
            wait=1
            # parity=serial.PARITY_NONE,
            # stopbits=serial.STOPBITS_ONE,
            # bytesize=serial.EIGHTBITS,
        )
        # if setup:
        #     self.setup_sensor()

        self._histogram = VL53L8CHHistogram(
            # self.num_channels,
            # self.active_channels_per_subcapture,
            # self.spad_id
        )
        # self._object = TMF8828Object()
        self._debug = debug

        if init:
            # self.queue = BlockingDeque(maxlen=5)
            # self.stop_thread = threading.Event()
            # self.has_started = threading.Event()
            # self.start_capture_event = threading.Event()
            # self._start_background_capture()

            self.initialize()

        self._resolution = 4, 4
        self._num_pixels = self._get_num_pixels()

        self._initialized = True

    def _get_num_pixels(self) -> int:
        return self._resolution[0] * self._resolution[1]

    def _get_num_channels(self) -> int:
        # Channels 0-9, including calibration channel
        return 10

    def _get_active_channels_per_subcapture(self) -> list[int]:
        return [9]

    def initialize(self):
        pass
        # get_logger().info("Initializing sensor...")

        # self.write("h")
        # self.wait_for_start_talk()
        # self.wait_for_stop_talk()

        # get_logger().info("Sensor initialized")

    def setup_sensor(self) -> None:
        pass
        # get_logger().info("Setting up sensor...")

        # # Reset the sensor
        # self.write("d")
        # self.wait_for_start_talk()
        # self.wait_for_stop_talk()

        # if self.spad_id in [SPADID.ID6, SPADID.ID7]:  # 3x3, 4x4
        #     self.write("o")  # Switch to TMF882x mode
        #     self.wait_for_start_talk()
        #     self.wait_for_stop_talk()
        #     self.write("E")
        #     self.wait_for_start_talk()
        #     self.wait_for_stop_talk()
        #     if self.spad_id == SPADID.ID7:  # 4x4
        #         self.write("c")  # Move to the next configuration
        #         self.wait_for_start_talk()
        #         self.wait_for_stop_talk()
        # elif self.spad_id == SPADID.ID15:  # 8x8
        #     self.write("e")
        #     self.wait_for_start_talk()
        #     self.wait_for_stop_talk()
        # else:
        #     raise ValueError(f"Unsupported mode: {self.spad_id}")

        # self.write("z")
        # self.wait_for_stop_talk()

        # get_logger().info("Sensor setup complete")

    def _start_background_capture(self):
        """Starts the background thread to initialize the camera and capture images."""
        self.thread = threading.Thread(target=self._background_capture)
        self.thread.start()

    def _background_capture(self):
        """Initializes the camera, continuously captures histograms and stores them in the queue."""
        get_logger().info(f"Starting background capture for spad")
        while not self.stop_thread.is_set():
            # Wait until capture is started
            self.start_capture_event.wait()
            try:
                self.has_started.set()

                while not self.stop_thread.is_set():
                    # get_logger().info(f"Sample {len(histograms) + 1}/{num_samples}")

                    began_read = False
                    # count = 0
                    while not self._histogram.has_data:
                        line = self.read()
                        get_logger().debug(line)
                        try:
                            line = line.decode("utf-8").replace("\n", "")
                            # data = data.decode("utf-8").replace("\r", "").replace("\n", "")
                        except UnicodeDecodeError:
                            get_logger().error("Error decoding data")
                            continue
                        # row = data.split(",")
                        # print(f'line: {line}')

                        if line[:10] == "Print data":
                            began_read = True
                            # print("began read")
                            continue
                        
                        if began_read:
                            if len(line) > 0 and line[0] != "":
                                tokens = []
                                for raw_token in line.split(","):
                                    token = raw_token.strip()
                                    if len(token) > 0:
                                        tokens.append(token)
                                self._histogram.process(tokens)
                        
                        # count += 1
                        # if count >= 50:
                        #     break

                    # histograms.append(self._histogram.get_data())

                    # Store histogram in queue
                    get_logger().info("Storing histogram in queue")
                    self.queue.append(self._histogram.get_data())
            except Exception as ex:
                get_logger().error(f"Camera error: {ex}")
            finally:
                # Stop the pipeline and reset events
                get_logger().info(f"Stopping pipeline")
                # self.pipeline.stop()
                self.has_started.clear()
                self.start_capture_event.clear()

        get_logger().info(f"Background capture thread ending")

    def read(self) -> bytes:
        read_line = self._arduino.readline()
        if self._debug and len(read_line) > 0:
            if read_line[0] != b'#'[0]:
                # print debug lines
                try:
                    read_line_str = read_line.decode("utf-8").replace("\r", "").replace("\n", "")
                    if len(read_line_str) > 5:
                        if read_line_str[:5] == "Debug":
                            get_logger().info(read_line_str)
                    # get_logger().info(read_line)
                except UnicodeDecodeError:
                    get_logger().error(read_line)

        return read_line

    def write(self, data: str) -> None:
        get_logger().debug(f"Writing {data}...")
        padded_str = data.ljust(10, " ")
        self._arduino.write(padded_str)
        # time.sleep(0.05)

    def change_num_bins(self, num_bins: int) -> None:
        get_logger().info(f"Changing num_bins to {num_bins}")
        self.write(f"b{num_bins}")
        self._histogram = VL53L8CHHistogram()
    
    def change_resolution(self, resolution: int) -> None:
        get_logger().info(f"Changing resolution to {resolution}")
        self.write(f"r{resolution}")
        if resolution == 1:
            self._resolution = 8, 8
        else:
            self._resolution = 4, 4
        self._num_pixels = self._get_num_pixels()

    # def wait_for_start_talk(self) -> bytes:
    #     """Wait until Arduino starts talking."""
    #     data = b""
    #     while len(data) == 0:
    #         data = self.read()
    #     return data

    # def wait_for_stop_talk(self) -> None:
    #     """Wait until Arduino stops talking."""
    #     data = b"0"
    #     while len(data) > 0:
    #         data = self.read()
    #         try:
    #             data_str = re.sub(r"[\r\n]", "", data.decode("utf-8").strip())
    #             get_logger().debug(data_str)
    #         except UnicodeDecodeError:
    #             get_logger().debug(data)

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
        # self.start_capture_event.set()
        # try:
        #     histograms = []

        #     while len(histograms) < num_samples:
        #         try:
        #             hist_item = self.queue.popleft()

        #             histograms.append(hist_item)
        #         except IndexError:
        #             continue  # Wait for more data if queue is empty

        #     if num_samples == 1:
        #         histograms = histograms[0] if histograms else None
        #     elif average:
        #         histograms = np.mean(histograms, axis=0)
        #     # print(histograms)
        #     return histograms
        # finally:
        #     self.start_capture_event.clear()
        #     self.has_started.clear()

        # flush serial buffer if needed
        BUFFER_FLUSH_THRESHOLD = 500
        if self._arduino.in_waiting > BUFFER_FLUSH_THRESHOLD:
            if self._debug:
                pass
                # get_logger().info(f"Flushed buffer: cleared {self._arduino.in_waiting} bytes")
            self._arduino.reset_input_buffer()

        histograms = []
        for _ in range(num_samples):
            # get_logger().info(f"Sample {len(histograms) + 1}/{num_samples}")

            began_read = False
            # count = 0
            while not self._histogram.has_data:
                line = self.read()
                get_logger().debug(line)
                try:
                    line = line.decode("utf-8").replace("\n", "")
                    # data = data.decode("utf-8").replace("\r", "").replace("\n", "")
                except UnicodeDecodeError:
                    get_logger().error("Error decoding data")
                    continue
                # row = data.split(",")
                # print(f'line: {line}')

                if line[:10] == "Print data":
                    began_read = True
                    # print("began read")
                    continue
                if line[:10] == "EndMessage":
                    num_hist_expected = int(line.split(" ")[1])
                    self._histogram._assemble_data(num_hist_expected)
                    continue
                elif began_read:
                    if len(line) > 0 and line[0] != "":
                        tokens = []
                        for raw_token in line.split(","):
                            token = raw_token.strip()
                            if len(token) > 0:
                                tokens.append(token)
                        self._histogram.process(tokens)
                
                # count += 1
                # if count >= 50:
                #     break
            histograms.append(self._histogram.get_data())

        if num_samples == 1:
            histograms = histograms[0] if histograms else None
        elif average:
            histograms = np.mean(histograms, axis=0)
        # print(histograms)
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
        return self._resolution

    @property
    def num_pixels(self) -> int:
        return self._num_pixels
    
    # @override
    # def close(self):
    #     """Stops the background capture thread"""
    #     self.stop_thread.set()  # Signal the background thread to stop
    #     self.start_capture_event.set()  # Unblock the thread if waiting
    #     if self.thread is not None:
    #         self.thread.join()  # Wait for the thread to finish
    #         self.thread = None
