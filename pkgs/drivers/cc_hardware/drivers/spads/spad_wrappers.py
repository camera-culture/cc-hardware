from typing import Any

import numpy as np

from cc_hardware.drivers.spads import SPADDataType, SPADSensor, SPADSensorConfig
from cc_hardware.drivers.spads.pkl import PklSPADSensor, PklSPADSensorConfig
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

    def __init__(self, config: SPADWrapperConfig, **kwargs):
        super().__init__(config)

        self._sensor = SPADSensor.create_from_config(config.wrapped, **kwargs)

    def reset(self, **kwargs) -> None:
        """Resets the sensor configuration to its initial state."""
        super().reset(**kwargs)
        self._sensor.reset(**kwargs)

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

    @property
    def unwrapped(self) -> SPADSensor:
        """Returns the underlying SPAD sensor."""
        return self._sensor

    def close(self):
        super().close()
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
    def update(self, **kwargs) -> None:
        super().update(**kwargs)

        if self.config.merge_rows and self.config.merge_cols:
            self.config.merge_all = True
        if self.config.merge_all:
            self.config.merge_rows = False
            self.config.merge_cols = False

    def accumulate(self, num_samples: int = 1, **kwargs):
        data = super().accumulate(num_samples=num_samples, **kwargs)

        if num_samples == 1:
            data = [data]

        for _data in data:
            if SPADDataType.HISTOGRAM in self.config.data_type:
                histograms = _data[SPADDataType.HISTOGRAM]
                _data[SPADDataType.HISTOGRAM] = self._merge(histograms)
            if SPADDataType.DISTANCE in self.config.data_type:
                distances = _data[SPADDataType.DISTANCE]
                _data[SPADDataType.DISTANCE] = self._merge(distances)
            if SPADDataType.POINT_CLOUD in self.config.data_type:
                point_clouds = _data[SPADDataType.POINT_CLOUD]
                _data[SPADDataType.POINT_CLOUD] = self._merge(point_clouds, axis=0)
            if SPADDataType.RAW in self.config.data_type:
                raise ValueError(
                    "SPADMergeWrapper does not support raw data type. "
                    "Please use a different wrapper or remove the raw data type."
                )

        if num_samples == 1:
            data = data[0]

        return data

    def _merge(self, data: np.ndarray, axis: int | None = None) -> np.ndarray:
        """Merges the data based on the configuration."""
        if axis is not None:
            return np.sum(data, axis=axis, keepdims=True)

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
    def __init__(self, config: SPADMovingAverageWrapperConfig, **kwargs):
        super().__init__(config, **kwargs)

        self._data: dict[SPADDataType, list[np.ndarray]] = {}

    def reset(self) -> None:
        super().reset()
        self._data.clear()

    def update(self, **kwargs) -> bool:
        if not super().update(**kwargs):
            return

        # Clear the accumulated data when the configuration is updated
        print("Clearing accumulated data for moving average wrapper.")
        self._data.clear()

        return True

    def accumulate(self, num_samples: int = 1, **kwargs):
        assert num_samples == 1
        data = super().accumulate(num_samples, **kwargs)

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
        moving_average = np.average(self._data[data_type], axis=0)
        return moving_average


# =============================================================================


@config_wrapper
class SPADBackgroundRemovalWrapperConfig(SPADWrapperConfig):
    """Configuration for SPAD sensor background removal wrapper. Note this only removes
    the background from the histogram.

    Args:
        pkl_spad (PklSPADSensorConfig): The configuration for the wrapped PklSPAD
            sensor.

        clip (bool): Whether to clip the histogram data to zero after background
            removal. If True, negative values in the histogram will be set to zero.

        remove_background (bool): Whether to remove the background from the data.
    """

    pkl_spad: PklSPADSensorConfig

    clip: bool = True

    remove_background: bool = True
    remove_background_setting: BoolSetting = BoolSetting.default_factory(
        title="Remove Background",
        value=True,
    )

    @property
    def settings(self) -> dict[str, Any]:
        settings = self.wrapped.settings
        settings["remove_background"] = self.remove_background_setting
        return settings


class SPADBackgroundRemovalWrapper(SPADWrapper[SPADBackgroundRemovalWrapperConfig]):
    def __init__(self, config: SPADBackgroundRemovalWrapperConfig, **kwargs):
        super().__init__(config, **kwargs)

        self._pkl_spad: PklSPADSensor = PklSPADSensor.create_from_config(
            config.pkl_spad
        )
        assert (
            SPADDataType.HISTOGRAM in self._pkl_spad.config.data_type
        ), "PklSPADSensor must have histogram data type for background removal."
        assert self._pkl_spad.config.num_bins == self.config.wrapped.num_bins, (
            "PklSPADSensor num_bins must match the wrapped sensor num_bins. "
            f"PklSPADSensor num_bins: {self._pkl_spad.config.num_bins}, "
            f"Wrapped sensor num_bins: {self.config.wrapped.num_bins}"
        )
        assert self._pkl_spad.config.width == self.config.wrapped.width, (
            "PklSPADSensor width must match the wrapped sensor width. "
            f"PklSPADSensor width: {self._pkl_spad.config.width}, "
            f"Wrapped sensor width: {self.config.wrapped.width}"
        )
        assert self._pkl_spad.config.height == self.config.wrapped.height, (
            "PklSPADSensor height must match the wrapped sensor height. "
            f"PklSPADSensor height: {self._pkl_spad.config.height}, "
            f"Wrapped sensor height: {self.config.wrapped.height}"
        )
        assert self._pkl_spad.config.start_bin == self.config.wrapped.start_bin, (
            "PklSPADSensor start_bin must match the wrapped sensor start_bin. "
            f"PklSPADSensor start_bin: {self._pkl_spad.config.start_bin}, "
            f"Wrapped sensor start_bin: {self.config.wrapped.start_bin}"
        )

        self._background: np.ndarray | None = None
        self._initialize_background()

    def _initialize_background(self):
        """Initializes the background data from the PklSPAD sensor."""
        histograms = []
        for i in range(1, len(self._pkl_spad.handler)):
            entry = self._pkl_spad.accumulate(index=1)
            assert (
                SPADDataType.HISTOGRAM in entry
            ), "PklSPADSensor must have histogram data type for background removal."
            histograms.append(entry[SPADDataType.HISTOGRAM])
        assert histograms, "No histogram data found in PklSPAD sensor."

        self._background = np.mean(histograms, axis=0)

    def update(self, **kwargs) -> bool:
        if not super().update(**kwargs):
            return False

        if "remove_background" in kwargs:
            self.config.remove_background = kwargs["remove_background"]

        return True

    def accumulate(self, num_samples: int = 1, **kwargs):
        data = super().accumulate(num_samples=num_samples, **kwargs)

        if self.config.remove_background and self._background is not None:
            samples = [data] if num_samples == 1 else data
            for sample in samples:
                hist = sample[SPADDataType.HISTOGRAM]
                hist -= self._background
                if self.config.clip:
                    np.clip(hist, a_min=0, a_max=None, out=hist)
                sample[SPADDataType.HISTOGRAM] = hist
            data = samples[0] if num_samples == 1 else samples

        return data

    def close(self):
        """Closes the PklSPAD sensor."""
        super().close()

        if hasattr(self, "_pkl_spad"):
            self._pkl_spad.close()


# =============================================================================


@config_wrapper
class SPADScalingWrapperConfig(SPADWrapperConfig):
    """Configuration for SPAD sensor scaling wrapper.
    This wrapper scales the histogram data by a given factor.

    Args:
        scale (float): The factor by which to scale the histogram.
    """

    scale: int = 1
    scale_setting: RangeSetting = RangeSetting.default_factory(
        title="Histogram Scale", min=0, max=100, value=II("..scale")
    )

    @property
    def settings(self) -> dict[str, Any]:
        settings = self.wrapped.settings
        settings["scale"] = self.scale_setting
        return settings


class SPADScalingWrapper(SPADWrapper[SPADScalingWrapperConfig]):
    """
    A wrapper class for SPAD sensors that scales the accumulated histogram data.
    """

    def __init__(self, config: SPADScalingWrapperConfig, **kwargs):
        super().__init__(config, **kwargs)

        assert (
            SPADDataType.HISTOGRAM in self._sensor.config.data_type
        ), "SPADScalingWrapper requires the wrapped sensor to have histogram data type."

    def update(self, **kwargs) -> bool:
        if not super().update(**kwargs):
            return False

        if "scale" in kwargs:
            self.config.scale = kwargs["scale"]

        return True

    def accumulate(self, *args, **kwargs):
        """
        Accumulates data from the wrapped sensor and scales the histogram.
        """
        data = super().accumulate(*args, **kwargs)

        if self.config.scale == 1.0:  # No scaling needed
            return data

        histogram_data = data[SPADDataType.HISTOGRAM]
        scaled_histogram = histogram_data * self.config.scale
        data[SPADDataType.HISTOGRAM] = scaled_histogram

        return data

# =============================================================================

# @config_wrapper
# class SPADSubpixelSampleWrapperConfig(SPADWrapperConfig):
#     pass

# class SPADSubpixelSampleWrapper(SPADWrapper[SPADSubpixelSampleWrapper]):
#     pass