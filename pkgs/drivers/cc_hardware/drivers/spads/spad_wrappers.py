from typing import Any

import numpy as np

from cc_hardware.drivers.spads import SPADDataType, SPADSensor, SPADSensorConfig
from cc_hardware.utils import II, config_wrapper
from cc_hardware.utils.setting import BoolSetting, RangeSetting


@config_wrapper
class SPADWrapperConfig(SPADSensorConfig):
    """Configuration for SPAD sensor wrapper.

    Args:
        wrapped (SPADSensorConfig): The configuration for the wrapped sensor.
    """

    wrapped: SPADSensorConfig

    data_type: SPADDataType = II(".wrapped.data_type")
    height: int = II(".wrapped.height")
    width: int = II(".wrapped.width")
    num_bins: int = II(".wrapped.num_bins")
    fovx: float = II(".wrapped.fovx")
    fovy: float = II(".wrapped.fovy")
    timing_resolution: float = II(".wrapped.timing_resolution")
    subsample: int = II(".wrapped.subsample")
    start_bin: int = II(".wrapped.start_bin")

    @property
    def settings(self) -> dict[str, Any]:
        """Retrieves the wrapped sensor settings."""
        return self.wrapped.settings


class SPADWrapper[T: SPADWrapperConfig](SPADSensor[T]):
    """
    A wrapper class for SPAD sensors that provides additional functionality and
    abstraction. This class is designed to wrap an existing SPAD sensor and expose
    additional methods and properties to simplify sensor management and data
    collection.

    Args:
        config (SPADWrapperConfig): The configuration for the sensor wrapper.
    """

    def __init__(self, config: SPADWrapperConfig):
        super().__init__(config)

        self._sensor = config.wrapped
        if not isinstance(config.wrapped, SPADSensor):
            self._sensor = SPADSensor.create_from_config(config.wrapped)

    def accumulate(self, *args, **kwargs):
        return self._sensor.accumulate(*args, **kwargs)

    @property
    def num_bins(self) -> int:
        return self._sensor.num_bins

    @property
    def resolution(self) -> tuple[int, int]:
        return self._sensor.resolution

    @property
    def is_okay(self) -> bool:
        return self._sensor.is_okay

    def close(self):
        if hasattr(self, "_sensor"):
            self._sensor.close()

    def calibrate(self) -> bool:
        return self._sensor.calibrate()

    def update(self, **kwargs) -> bool:
        return self._sensor.update(**kwargs) or super().update(**kwargs)


# =============================================================================


@config_wrapper
class SPADMergeWrapperConfig(SPADWrapperConfig):
    """Configuration for SPAD sensor merge wrapper.

    Args:
        merge_rows (bool): Whether to merge the rows of the histogram.
        merge_cols (bool): Whether to merge the columns of the histogram.
        merge_all (bool): Whether to merge all histogram data. If True, merge_rows and
            merge_cols are ignored.
    """

    merge_rows: bool = False
    merge_cols: bool = False
    merge_all: bool = False

    merge_rows_setting: BoolSetting = BoolSetting.default_factory(
        title="Merge Rows", value=II("..merge_rows")
    )
    merge_cols_setting: BoolSetting = BoolSetting.default_factory(
        title="Merge Columns", value=II("..merge_cols")
    )
    merge_all_setting: BoolSetting = BoolSetting.default_factory(
        title="Merge All", value=II("..merge_all")
    )

    @property
    def settings(self) -> dict[str, Any]:
        settings = self.wrapped.settings
        settings["merge_rows"] = self.merge_rows_setting
        settings["merge_cols"] = self.merge_cols_setting
        settings["merge_all"] = self.merge_all_setting
        return settings


class SPADMergeWrapper(SPADWrapper[SPADMergeWrapperConfig]):
    def __init__(self, config: SPADMergeWrapperConfig):
        super().__init__(config)

    def update(self, **kwargs) -> None:
        super().update(**kwargs)

        if self.config.merge_rows and self.config.merge_cols:
            self.config.merge_all = True
        if self.config.merge_all:
            self.config.merge_rows = False
            self.config.merge_cols = False

    def accumulate(self, *args, **kwargs):
        data = super().accumulate(*args, **kwargs)

        if SPADDataType.HISTOGRAM in self.config.data_type:
            histograms = data[SPADDataType.HISTOGRAM]
            data[SPADDataType.HISTOGRAM] = self._merge(histograms)
        if SPADDataType.DISTANCE in self.config.data_type:
            distances = data[SPADDataType.DISTANCE]
            data[SPADDataType.DISTANCE] = self._merge(distances)
        if SPADDataType.POINT_CLOUD in self.config.data_type:
            point_clouds = data[SPADDataType.POINT_CLOUD]
            data[SPADDataType.POINT_CLOUD] = self._merge(point_clouds)
        if SPADDataType.RAW in self.config.data_type:
            raise ValueError(
                "SPADMergeWrapper does not support raw data type. "
                "Please use a different wrapper or remove the raw data type."
            )

        return data

    def _merge(self, data: np.ndarray) -> np.ndarray:
        """Merges the data based on the configuration."""
        if self.config.merge_rows:
            data = np.sum(data, axis=0, keepdims=True)
        if self.config.merge_cols:
            data = np.sum(data, axis=1, keepdims=True)
        if self.config.merge_all:
            data = np.sum(data, axis=(0, 1), keepdims=True)
        return data

    @property
    def resolution(self) -> tuple[int, int]:
        resolution = super().resolution
        if self.config.merge_rows:
            resolution = (1, resolution[1])
        if self.config.merge_cols:
            resolution = (resolution[0], 1)
        if self.config.merge_all:
            resolution = (1, 1)
        return resolution


# =============================================================================


@config_wrapper
class SPADMovingAverageWrapperConfig(SPADWrapperConfig):
    """Configuration for SPAD sensor moving average wrapper.

    Args:
        window_size (int): The size of the moving average window.
    """

    window_size: int

    window_size_setting: RangeSetting = RangeSetting.default_factory(
        title="Window Size", min=1, max=100, value=II("..window_size")
    )

    @property
    def settings(self) -> dict[str, Any]:
        settings = self.wrapped.settings
        settings["window_size"] = self.window_size_setting
        return settings


class SPADMovingAverageWrapper(SPADWrapper[SPADMovingAverageWrapperConfig]):
    def __init__(self, config: SPADMovingAverageWrapperConfig):
        super().__init__(config)

        self._data: dict[SPADDataType, list[np.ndarray]] = {}

    def update(self, **kwargs) -> bool:
        if not super().update(**kwargs):
            return

        # Clear the accumulated data when the configuration is updated
        print("Clearing accumulated data for moving average wrapper.")
        self._data.clear()

        return True

    def accumulate(self, *args, **kwargs):
        data = super().accumulate(*args, **kwargs)

        if SPADDataType.HISTOGRAM in self.config.data_type:
            data[SPADDataType.HISTOGRAM] = self._moving_average(
                data, SPADDataType.HISTOGRAM
            )
        if SPADDataType.DISTANCE in self.config.data_type:
            data[SPADDataType.DISTANCE] = self._moving_average(
                data, SPADDataType.DISTANCE
            )
        if SPADDataType.POINT_CLOUD in self.config.data_type:
            data[SPADDataType.POINT_CLOUD] = self._moving_average(
                data, SPADDataType.POINT_CLOUD
            )
        if SPADDataType.RAW in self.config.data_type:
            raise ValueError(
                "SPADMovingAverageWrapper does not support raw data type. "
                "Please use a different wrapper or remove the raw data type."
            )

        return data

    def _moving_average(
        self, data: dict[SPADDataType, np.ndarray], data_type: SPADDataType
    ) -> np.ndarray:
        """Calculates the moving average of the data."""

        self._data.setdefault(data_type, [])
        self._data[data_type].append(data[data_type].copy())
        if len(self._data[data_type]) > self.config.window_size:
            self._data[data_type].pop(0)
        moving_average = np.mean(self._data[data_type], axis=0)
        return moving_average
