"""Base class for Single-Photon Avalanche Diode (SPAD) sensors."""

from abc import abstractmethod
from enum import Flag, auto
from typing import Any

import numpy as np
from hydra_config.utils import HydraFlagWrapperMeta

from cc_hardware.drivers.sensor import Sensor, SensorConfig, SensorData
from cc_hardware.utils import config_wrapper
from cc_hardware.utils.constants import C

# ================


class SPADDataType(Flag, metaclass=HydraFlagWrapperMeta):
    """Enum for SPAD data types."""

    HISTOGRAM = auto()
    DISTANCE = auto()
    POINT_CLOUD = auto()
    RAW = auto()


@config_wrapper
class SPADSensorConfig(SensorConfig):
    """Configuration for SPAD sensors.

    Attributes:
        data_type (SPADDataType): The type of data the sensor will collect.
            Default is SPADDataType.HISTOGRAM.
    """

    data_type: SPADDataType = SPADDataType.HISTOGRAM

    height: int
    width: int
    num_bins: int
    fovx: float
    fovy: float
    timing_resolution: float
    start_bin: int = 0
    subsample: int = 1

    @property
    def num_pixels(self) -> int:
        """Returns the total number of pixels in the sensor."""
        return self.height * self.width


class SPADSensor[T: SPADSensorConfig](Sensor[T]):
    """
    An abstract base class for Single-Photon Avalanche Diode (SPAD) sensors, designed
    to manage histogram-based measurements. This class defines methods and properties
    related to collecting and analyzing histogram data.

    Inherits:
        Sensor: The base class for all sensors in the system.
    """

    def __init__(self, config: T):
        super().__init__(config)

        self._data: SPADSensorData[T]

    @abstractmethod
    def accumulate(
        self, num_samples: int = 1
    ) -> list[dict[SPADDataType, np.ndarray]] | dict[SPADDataType, np.ndarray]:
        """
        Accumulates the specified number of histogram samples from the sensor.

        Args:
            num_samples (int): The number of samples to accumulate into the histogram.
                The accumulation method (i.e. summing, averaging) may vary depending on
                the sensor. Defaults to 1.

        Returns:
            list[dict[SPADDataType, np.ndarray]] | dict[SPADDataType, np.ndarray]:
                The accumulated histogram data. The format may vary depending on the
                sensor and the number of samples.
        """
        pass

    @property
    def num_bins(self) -> int:
        """
        Returns the number of bins in the sensor's histogram. This indicates the
        number of discrete values or ranges that the sensor can measure. The total
        distance a sensor can measure is equal to the number of bins multiplied by
        the bin width.

        Returns:
            int: The total number of bins in the histogram.
        """
        return self._config.num_bins

    @num_bins.setter
    def num_bins(self, value: int):
        """
        Sets the number of bins in the sensor's histogram. This method allows the
        number of bins to be dynamically adjusted to match the sensor's configuration.

        Args:
            value (int): The new number of bins in the histogram.
        """
        self._data.reset()
        self.update(num_bins=value)

    @property
    def resolution(self) -> tuple[int, int]:
        """
        Returns the resolution of the sensor as a tuple (width, height). This indicates
        the spatial resolution of the sensor, where the width and height refer to the
        number of pixels or sampling points in the respective dimensions.

        Returns:
            tuple[int, int]: A tuple representing the sensor's resolution
                (width, height).
        """
        return self.config.width, self.config.height

    @resolution.setter
    def resolution(self, value: tuple[int, int]):
        """
        Sets the resolution of the sensor. This method allows the resolution to be
        dynamically adjusted to match the sensor's configuration.

        Args:
            value (tuple[int, int]): The new resolution of the sensor as a tuple
                (width, height).
        """
        self.update(width=value[0], height=value[1])


class SPADSensorData[T: SPADSensorConfig](SensorData):
    def __init__(self, config: T):
        super().__init__()

        self._config = config
        self._data: dict[SPADDataType, Any] = {}
        self._ready_data: dict[SPADDataType, np.ndarray] = {}
        self._data_type = config.data_type

    def reset(self):
        super().reset()
        self._data.clear()
        self._ready_data.clear()

    @property
    def has_data(self) -> bool:
        """Checks if the sensor has data available."""
        return bool(self._ready_data)

    def process(self, data: dict[SPADDataType, np.ndarray]) -> None:
        """
        Processes the incoming data and stores it in the appropriate format.

        Args:
            data (dict[SPADDataType, np.ndarray]): The incoming data to process.
        """
        self._data.update(data)
        self._ready_data.update(data)

    def get_data(
        self, *, verify_has_data: bool = True
    ) -> dict[SPADDataType, np.ndarray]:
        assert not verify_has_data or self.has_data, "No data available."

        data: dict[SPADDataType, np.ndarray] = {}
        if SPADDataType.HISTOGRAM in self._data_type:
            assert (
                SPADDataType.HISTOGRAM in self._ready_data
            ), "No histogram data available."
            data[SPADDataType.HISTOGRAM] = self._ready_data[
                SPADDataType.HISTOGRAM
            ].copy()
        if SPADDataType.DISTANCE in self._data_type:
            if SPADDataType.DISTANCE in self._data:
                data[SPADDataType.DISTANCE] = self._ready_data[
                    SPADDataType.DISTANCE
                ].copy()
            elif SPADDataType.HISTOGRAM in self._data:
                data[SPADDataType.DISTANCE] = self.calculate_distance(
                    self._ready_data[SPADDataType.HISTOGRAM].copy()
                )
            else:
                raise ValueError(
                    "No distance data available. "
                    "Please provide either distance or histogram data."
                )
        if SPADDataType.POINT_CLOUD in self._data_type:
            if SPADDataType.POINT_CLOUD in self._data:
                data[SPADDataType.POINT_CLOUD] = self._ready_data[
                    SPADDataType.POINT_CLOUD
                ].copy()
            elif SPADDataType.DISTANCE in self._data:
                data[SPADDataType.POINT_CLOUD] = self.calculate_point_cloud(
                    distances=self._ready_data[SPADDataType.DISTANCE].copy()
                )
            elif SPADDataType.HISTOGRAM in self._data:
                data[SPADDataType.POINT_CLOUD] = self.calculate_point_cloud(
                    histogram=self._ready_data[SPADDataType.HISTOGRAM].copy()
                )
            else:
                raise ValueError(
                    "No point cloud data available. "
                    "Please provide either point cloud, distance, or histogram data."
                )
        if SPADDataType.RAW in self._data_type:
            assert SPADDataType.RAW in self._data, "No raw data available."
            data[SPADDataType.RAW] = self._ready_data[SPADDataType.RAW].copy()

        # Now reset
        self.reset()

        return data

    def calculate_point_cloud(
        self,
        *,
        histogram: np.ndarray | None = None,
        distances: np.ndarray | None = None,
        subpixel_samples: int = 1,
        bilinear_interpolation: bool = False,
    ) -> np.ndarray:
        """
        Calculates the point cloud from histogram or precomputed distances.

        Args:
            histogram (np.ndarray): Histogram data, if distances not provided.
            distances (np.ndarray): Precomputed distances (mm).

        Keyword Args:
            subpixel_samples (int): Number of samples per pixel.
            bilinear_interpolation (bool): Whether to interpolate distances.

        Returns:
            np.ndarray: Point cloud (N, 3) in meters.
        """
        assert histogram is not None or distances is not None
        assert not (histogram is not None and distances is not None), (
            "Either histogram or distances must be provided, " "but not both."
        )

        # 1) Compute distances
        if distances is None:
            distances = self.calculate_distance(histogram)

        # 2) Compute angular resolution per pixel
        H, W = distances.shape
        px_x = np.radians(self._config.fovx) / W
        px_y = np.radians(self._config.fovy) / H

        # 3) Build point cloud
        grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        samples = np.linspace(
            0.5 / subpixel_samples, 1 - 0.5 / subpixel_samples, subpixel_samples
        )

        pts = []
        for sy in samples:
            for sx in samples:
                y_sub = grid_y + sy
                x_sub = grid_x + sx

                if bilinear_interpolation:
                    x0 = np.floor(x_sub).astype(int)
                    y0 = np.floor(y_sub).astype(int)
                    x1 = np.clip(x0 + 1, 0, W - 1)
                    y1 = np.clip(y0 + 1, 0, H - 1)
                    dx = x_sub - x0
                    dy = y_sub - y0

                    d = (
                        distances[y0, x0] * (1 - dx) * (1 - dy)
                        + distances[y0, x1] * dx * (1 - dy)
                        + distances[y1, x0] * (1 - dx) * dy
                        + distances[y1, x1] * dx * dy
                    )
                else:
                    d = distances[grid_y, grid_x]

                d = np.maximum(0.0, d)
                angle_x = y_sub * px_y - np.radians(self._config.fovy) / 2 - np.pi / 2
                angle_y = x_sub * px_x - np.radians(self._config.fovx) / 2

                x = (d * np.cos(angle_x)) / 1e3
                y = (d * np.sin(angle_y)) / 1e3
                z = d / 1e3
                pts.append(np.stack([x, y, z], axis=-1))

        return np.concatenate([p.reshape(-1, 3) for p in pts], axis=0)

    def calculate_distance(
        self, histogram: np.ndarray, *, window: int = 10, threshold: float = 0
    ) -> np.ndarray:
        """
        Calculates the distance from the histogram data.

        Args:
            histogram (np.ndarray): The histogram data.

        Keyword Args:
            window (int): The size of the window to use for distance calculation.
                Defaults to 10.

        Returns:
            np.ndarray: The calculated distance.
        """
        assert histogram.shape == (
            self._config.height,
            self._config.width,
            self._config.num_bins,
        ), f"Histogram shape mismatch. Expected {self._config.height, self._config.width, self._config.num_bins}, got {histogram.shape}."

        distances = np.zeros((self._config.height, self._config.width), dtype=float)

        for i in range(self._config.height):
            for j in range(self._config.width):
                idx = histogram[i, j, :].argmax()

                start = max(0, idx - window // 2)
                end = min(self._config.num_bins, idx + window // 2 + 1)
                bins = np.arange(start, end) + self._config.start_bin

                # Calculate a weighting term to be used to
                # window the distances to smooth them out
                weights = histogram[i, j, start:end].astype(float)
                if weights.sum() < threshold:
                    continue
                weights /= weights.sum()

                # Calculate the time-of-flight
                tof = bins * self._config.timing_resolution * self._config.subsample
                distances[i, j] = C * np.dot(weights, tof) / 2

        return distances
