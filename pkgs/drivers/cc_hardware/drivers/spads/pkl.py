"""SPAD sensor driver that loads pre-recorded data from a PKL file."""

from pathlib import Path
from copy import deepcopy

import numpy as np

from cc_hardware.drivers.spads.spad import SPADSensor, SPADSensorConfig
from cc_hardware.utils.file_handlers import PklReader
from cc_hardware.utils import get_logger, register
from cc_hardware.utils.config import config_wrapper

# ==================


@register
@config_wrapper
class PklSPADSensorConfig(SPADSensorConfig):
    instance: str = "PklSPADSensor"

    pkl_path: Path | str
    key: str = "histogram"
    resolution: tuple[int, int] | None = None
    merge: bool = True


@register
class PklSPADSensor(SPADSensor):
    """
    A fake SPAD sensor class that simulates sensor behavior by loading pre-recorded
    histogram data from a PKL file. This class inherits from SPADSensor and is used
    for testing or simulations without actual hardware.

    Inherits:
        SPADSensor: Base class for SPAD sensors that defines common methods and
            properties.

    Args:
        config (PklSPADSensorConfig): The configuration object for the fake sensor.
    """

    def __init__(self, config: PklSPADSensorConfig, index: int = 0):
        super().__init__(config)

        config.pkl_path = Path(config.pkl_path)
        assert config.pkl_path.exists(), f"PKL file {config.pkl_path} does not exist."
        self._handler = PklReader(config.pkl_path)
        assert len(self._handler) > 0, "No data found in PKL file."
        self._index = 0

        entry = self._handler.load(index)
        assert config.key in entry, f"Key '{config.key}' not found in data."
        if config.resolution is None:
            config.resolution = entry[config.key].shape[:-1]
            if config.merge:
                config.resolution = [1, 1]
            else:
                config.resolution = [3, 3]
            # assert len(config.resolution) == 2, "Invalid resolution shape."
        self._first_entry = deepcopy(entry)

    def reset(self, index: int = 0):
        """
        Resets the sensor state. This method is a no-op for this fake sensor.
        """
        self._index = index

    @property
    def config(self) -> PklSPADSensorConfig:
        """
        Returns the configuration object for the sensor.

        Returns:
            PklSPADSensorConfig: The sensor's configuration object.
        """
        return self._config

    @property
    def handler(self) -> PklReader:
        return self._handler

    def accumulate(
        self,
        num_samples: int = 1,
        *,
        average: bool = True,
        return_entry: bool = False,
        index: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, dict] | None:
        """
        Accumulates the specified number of histogram samples from the pre-recorded
        data.

        Args:
            num_samples (int): The number of samples to accumulate.

        Keyword Args:
            average (bool): Whether to average the accumulated samples. Defaults to
                True.
            return_entry (bool): Whether to return the loaded entry. Defaults to
                False.
            index (int): The index of the entry to load. If None, the next entry will
                be loaded. Defaults to None. Will set the index within the handler.

        Returns:
            np.ndarray | tuple[np.ndarray, dict] | None: The accumulated histogram
                data, or a tuple of the data and the loaded
        """
        if index is not None:
            self._index = index

        if self._index >= len(self._handler):
            get_logger().error("No more data available.")
            return None

        histograms = []
        for _ in range(num_samples):
            try:
                entry = self._handler.load(self._index)
                self._index += 1
            except StopIteration:
                get_logger().error("No more data available.")
                break

            histogram = entry[self.config.key]
            if self.config.merge:
                histogram = np.expand_dims(np.mean(histogram, axis=0), axis=0)
            histograms.append(histogram)
        else:
            histograms = np.array(histograms)
            if average and len(histograms) > 1:
                histograms = np.mean(histograms, axis=0)

            if not self.config.merge:
                histograms = np.squeeze(histograms)
            if return_entry:
                return histograms, entry
            return histograms

        return None

    @property
    def num_bins(self) -> int:
        """
        Returns the number of bins in the sensor's histogram.

        Returns:
            int: The number of bins in the histogram.
        """
        return self._first_entry[self._config.key].shape[-1]

    @property
    def resolution(self) -> tuple[int, int]:
        """
        Returns the resolution of the sensor as a tuple (width, height).

        Returns:
            tuple[int, int]: The resolution (width, height) of the sensor.
        """
        return self.config.resolution

    @property
    def is_okay(self) -> bool:
        return len(self._handler) > self._index

    def close(self) -> None:
        """
        Closes the sensor connection. This method is a no-op for this fake sensor.
        """
        pass
