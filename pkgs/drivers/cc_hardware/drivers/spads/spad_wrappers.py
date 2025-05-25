from typing import Any

import numpy as np

from cc_hardware.drivers.spads import SPADSensor, SPADSensorConfig
from cc_hardware.utils import II, config_wrapper
from cc_hardware.utils.setting import BoolSetting, RangeSetting


@config_wrapper
class SPADWrapperConfig(SPADSensorConfig):
    """Configuration for SPAD sensor wrapper.

    Args:
        wrapped (SPADSensorConfig): The configuration for the wrapped sensor.
    """

    wrapped: SPADSensorConfig

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
        histograms = self._sensor.accumulate(*args, **kwargs)

        return histograms

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

    def update(self, **kwargs) -> None:
        self._sensor.update(**kwargs)
        super().update(**kwargs)


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
        histograms = super().accumulate(*args, **kwargs)

        histograms = np.reshape(histograms, (*super().resolution, -1))
        if self.config.merge_rows:
            histograms = np.expand_dims(np.sum(histograms, axis=0), axis=0)
        if self.config.merge_cols:
            histograms = np.expand_dims(np.sum(histograms, axis=1), axis=1)
        if self.config.merge_all:
            histograms = np.expand_dims(np.sum(histograms, axis=(0, 1)), axis=0)

        histograms = np.reshape(histograms, (-1, histograms.shape[-1]))
        return histograms

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
        title="Window Size", min=1, max=1000, value=10  # II("..window_size")
    )

    @property
    def settings(self) -> dict[str, Any]:
        settings = self.wrapped.settings
        settings["window_size"] = self.window_size_setting
        return settings


class SPADMovingAverageWrapper(SPADWrapper[SPADMovingAverageWrapperConfig]):
    def __init__(self, config: SPADMovingAverageWrapperConfig):
        super().__init__(config)

        self._histograms = []

    def accumulate(self, *args, **kwargs):
        histograms = super().accumulate(*args, **kwargs)

        self._histograms.append(histograms)
        if len(self._histograms) > self.config.window_size:
            self._histograms.pop(0)

        moving_average = np.mean(self._histograms, axis=0)
        return moving_average
