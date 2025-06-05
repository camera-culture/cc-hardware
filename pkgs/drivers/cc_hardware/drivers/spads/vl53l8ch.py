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
from enum import Enum
from pathlib import Path

import numpy as np
import pkg_resources

from cc_hardware.drivers.safe_serial import SafeSerial
from cc_hardware.drivers.spads.spad import (
    SPADDataType,
    SPADSensor,
    SPADSensorConfig,
    SPADSensorData,
)
from cc_hardware.utils import II, config_wrapper, get_logger
from cc_hardware.utils.setting import BoolSetting, OptionSetting, RangeSetting, Setting

# ===============


class RangingMode(Enum):
    """
    Enumeration for the ranging mode of the VL53L8CH sensor.
    """

    CONTINUOUS = 1
    AUTONOMOUS = 3


@config_wrapper
class VL53L8CHConfig(SPADSensorConfig):
    """
    Configuration parameters for the VL53L8CH sensor.

    Attributes:
        port (str): Serial port for the sensor.

        resolution (int): Sensor resolution (uint16_t).
        ranging_mode (RangingMode): Ranging mode (uint16_t).
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

        add_back_ambient (bool): Flag to add back ambient light. The VL53L8CH sensor
            will do some preprocessing on the device and remove a pre-calculated ambient
            light value from the histogram data. In the histogram returned to the user,
            the ambient value removed from the histogram is provided. This flag enables
            the user to add this value back to the histogram data (as if it was never
            removed).
    """

    port: str | None = None

    fovx: float = 45.0
    fovy: float = 45.0
    timing_resolution: float = 250e-12

    ranging_mode: RangingMode  # uint16_t
    ranging_frequency_hz: int  # uint16_t
    integration_time_ms: int  # uint16_t
    agg_start_x: int  # uint16_t
    agg_start_y: int  # uint16_t
    agg_merge_x: int  # uint16_t
    agg_merge_y: int  # uint16_t

    add_back_ambient: bool

    # TODO: Resolution isn't supported right now. Need a way to set height/width from
    # single setting.
    # resolution_setting: OptionSetting = OptionSetting.default_factory(
    #     value=II("..resolution"), options=['4x4', '8x8'], title="Resolution"
    # )
    ranging_mode_setting: OptionSetting = OptionSetting.from_enum(
        enum=RangingMode, default=II("..ranging_mode"), title="Ranging Mode"
    )
    ranging_frequency_hz_setting: RangeSetting = RangeSetting.default_factory(
        value=II("..ranging_frequency_hz"),
        min=1,
        max=30,
        title="Ranging Frequency (Hz)",
    )
    integration_time_ms_setting: RangeSetting = RangeSetting.default_factory(
        value=II("..integration_time_ms"),
        min=10,
        max=1000,
        title="Integration Time (ms)",
    )
    num_bins_setting: RangeSetting = RangeSetting.default_factory(
        value=II("..num_bins"), min=1, max=32, title="Number of Bins"
    )
    add_back_ambient_setting: BoolSetting = BoolSetting.default_factory(
        value=II("..add_back_ambient"),
        title="Add Back Ambient Light",
    )
    start_bin_setting: RangeSetting = RangeSetting.default_factory(
        value=II("..start_bin"),
        min=0,
        max=128,
        title="Start Bin",
    )

    def pack(self) -> bytes:
        """
        Packs the sensor configuration into a byte structure.

        Returns:
            bytes: Packed configuration data.
        """
        return struct.pack(
            "<13H",
            self.height * self.width,
            self.ranging_mode.value,
            self.ranging_frequency_hz,
            self.integration_time_ms,
            self.start_bin,
            self.num_bins,
            self.subsample,
            self.agg_start_x,
            self.agg_start_y,
            self.agg_merge_x,
            self.agg_merge_y,
            self.width,
            self.height,
        )

    @property
    def settings(self) -> dict[str, Setting]:
        """
        Retrieves the configuration settings for the sensor.

        Returns:
            dict[str, Setting]: Configuration settings.
        """
        return {
            # "resolution": self.resolution_setting,
            "ranging_mode": self.ranging_mode_setting,
            "ranging_frequency_hz": self.ranging_frequency_hz_setting,
            "integration_time_ms": self.integration_time_ms_setting,
            "num_bins": self.num_bins_setting,
            "add_back_ambient": self.add_back_ambient_setting,
            "start_bin": self.start_bin_setting,
        }


@config_wrapper
class VL53L8CHSharedConfig(VL53L8CHConfig):
    """
    Shared sensor configuration with default settings.

    Inherits from SensorConfig and provides default values for common parameters.
    """

    ranging_mode: RangingMode = RangingMode.CONTINUOUS
    integration_time_ms: int = 10
    start_bin: int = 0
    subsample: int = 16
    agg_start_x: int = 0
    agg_start_y: int = 0
    agg_merge_x: int = 1
    agg_merge_y: int = 1

    add_back_ambient: bool = False


@config_wrapper
class VL53L8CHConfig4x4(VL53L8CHSharedConfig):
    """
    Sensor configuration for a 4x4 resolution.
    """

    height: int = 4
    width: int = 4
    num_bins: int = 8
    ranging_frequency_hz: int = 30


@config_wrapper
class VL53L8CHConfig8x8(VL53L8CHSharedConfig):
    """
    Sensor configuration for an 8x8 resolution.
    """

    height: int = 8
    width: int = 8
    num_bins: int = 8
    ranging_frequency_hz: int = 15


# ===============


class VL53L8CHData(SPADSensorData[VL53L8CHConfig]):
    """
    Processes and stores both histogram and target data from the VL53L8CH sensor.

    This class handles the accumulation and processing of histogram bins
    and per-pixel target information, keeping them aligned by pixel index.
    """

    def process(self, row: list[str]) -> bool:
        """
        Processes a row of data, routing to histogram or target handlers.

        Args:
            row (list[str]): A row of string values from sensor output.

        Returns:
            bool: True if processing succeeds, False otherwise.
        """

        if not self._process_histogram(row):
            return False
        if len(self._data[SPADDataType.HISTOGRAM]) == self._config.num_pixels:
            try:
                self._finalize()
            except ValueError as e:
                get_logger().error(f"Error finalizing data: {e}")
                return False
        return True

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
        histogram = self._data.setdefault(SPADDataType.HISTOGRAM, {})
        distance = self._data.setdefault(SPADDataType.DISTANCE, {})

        try:
            idx = int(row[0])
            if idx in histogram:
                get_logger().error(f"Duplicate histogram for pixel {idx}")
                return False

            ambient = float(row[1]) if self._config.add_back_ambient else 0.0
            bins = np.array([float(v) + ambient for v in row[3:]])

            histogram[idx] = np.clip(bins, 0, None)
            distance[idx] = float(row[2])
            return True
        except (ValueError, IndexError) as e:
            get_logger().error(f"Invalid histogram formatting: {e}")
            return False

    def _finalize(self) -> None:
        """
        Finalizes data when all pixels processed, storing combined output.
        """
        histogram = self._data[SPADDataType.HISTOGRAM]
        self._ready_data[SPADDataType.HISTOGRAM] = np.array(
            [histogram[i] for i in sorted(histogram)]
        ).reshape(
            self._config.height,
            self._config.width,
            self._config.num_bins,
        )

        distance = self._data[SPADDataType.DISTANCE]
        self._ready_data[SPADDataType.DISTANCE] = np.array(
            [distance[i] for i in sorted(distance)]
        ).reshape(self._config.height, self._config.width)


# ===============


class VL53L8CHSensor(SPADSensor[VL53L8CHConfig]):
    """
    Main sensor class for the VL53L8CH time-of-flight sensor.

    This class handles communication with the sensor, configuration,
    data acquisition, and data processing.

    Args:
        config (VL53L8CHConfig): Configuration parameters for the sensor.

    Keyword Args:
        **kwargs: Additional configuration parameters to update.

    Attributes:
        SCRIPT (Path): Path to the sensor's makefile script.
        BAUDRATE (int): Serial communication baud rate.
    """

    SCRIPT: Path = Path(
        pkg_resources.resource_filename(
            "cc_hardware.drivers",
            str(Path("data") / "vl53l8ch" / "build" / "makefile"),
        )
    )
    BAUDRATE: int = 2_250_000

    def __init__(
        self,
        config: VL53L8CHConfig,
        **kwargs,
    ):
        """
        Initializes the VL53L8CHSensor instance.

        Args:
            config (VL53L8CHConfig): Configuration parameters for the sensor.

        Keyword Args:
            **kwargs: Configuration parameters to update. Keys must match
                the fields of SensorConfig.
        """
        super().__init__(config)
        self._config = config
        self._data = VL53L8CHData(config)

        # inter-process communication queues/events
        self._queue = multiprocessing.Queue(
            maxsize=self.config.height * self.config.width * 4
        )
        self._write_queue = multiprocessing.Queue(maxsize=10)
        self._initialized_event = multiprocessing.Event()
        self._stop_event = multiprocessing.Event()

        self.update(**kwargs)

        self._reader_process = multiprocessing.Process(
            target=self._read_serial_background,
            args=(
                config.port,
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
            serial_conn = SafeSerial.create(
                port=port, baudrate=baudrate, one=True, timeout=1
            )
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
                line = serial_conn.readline()
                if line:
                    # Put the line into the queue without blocking
                    try:
                        queue.put_nowait(line)
                    except multiprocessing.queues.Full:
                        # Queue is full; discard the line to prevent blocking
                        pass

                # =====
                # WRITE
                try:
                    config_data = write_queue.get_nowait()
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
        Updates the sensor configuration with provided keyword arguments. If there are
        any changes given via the kwargs or in the settings, the configuration is sent
        to the sensor.

        Args:
            **kwargs: Configuration parameters to update. Keys must match
                the fields of SensorConfig.
        """
        if not super().update(**kwargs):
            return

        # Send the configuration to the sensor
        try:
            self._write_queue.put(self.config.pack())
        except multiprocessing.queues.Full:
            get_logger().error("Failed to send configuration to sensor")

    def accumulate(
        self, num_samples: int = 1
    ) -> list[dict[SPADDataType, np.ndarray]] | dict[SPADDataType, np.ndarray]:
        """
        Accumulates histogram and target data from the sensor.

        Args:
            num_samples (int): Number of samples to accumulate.

        Returns:
        """
        samples = []
        for _ in range(num_samples):
            self._data.reset()
            began = False
            while not self._data.has_data:
                try:
                    raw: bytes = self._queue.get(timeout=1)
                except multiprocessing.queues.Empty:
                    continue

                try:
                    line: str = raw.decode("utf-8").strip()
                    get_logger().debug(f"Processing line: {line}")
                except UnicodeDecodeError:
                    get_logger().error("Error decoding data")
                    continue

                if line.startswith("D"):
                    began = True
                    continue

                if began:
                    tokens = [tok.strip() for tok in line.split(" ") if tok.strip()]
                    if not self._data.process(tokens):
                        get_logger().error(f"Error processing row: {tokens}")
                        self._data.reset()
                        began = False
                        continue

            # Has data!
            samples.append(self._data.get_data())

        if num_samples == 1:
            return samples[0]
        return samples

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
        if not hasattr(self, "_stop_event"):
            return

        try:
            self._stop_event.set()
            if not self._initialized_event.is_set():
                return
        except AttributeError:
            get_logger().debug(
                f"{self.__class__.__name__} already closed or not initialized."
            )
            return

        # Signal the reader process to stop
        self._reader_process.join()
