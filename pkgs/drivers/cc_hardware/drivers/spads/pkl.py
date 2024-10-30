from pathlib import Path

import numpy as np

from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.writers import PklWriter
from cc_hardware.utils.registry import register
from cc_hardware.drivers.spads import SPADSensor


@register
class PklSPADSensor(SPADSensor):
    """
    A fake SPAD sensor class that simulates sensor behavior by loading pre-recorded
    histogram data from a PKL file. This class inherits from SPADSensor and is used
    for testing or simulations without actual hardware.

    Inherits:
        SPADSensor: Base class for SPAD sensors that defines common methods and properties.

    Attributes:
        _pkl_path (Path): Path to the PKL file containing pre-recorded data.
        _data (list[dict]): A list of entries loaded from the PKL file, each entry containing
            a histogram.
        _data_iterator (iterator): An iterator over the loaded data entries.
        _bin_width (float): The width of each bin in the histogram.
        _resolution (tuple[int, int]): The spatial resolution of the sensor.
    """

    def __init__(
        self, pkl_path: Path | str, *, bin_width: float, resolution: tuple[int, int]
    ):
        """
        Initializes the PklSPADSensor with the path to the PKL file, bin width, and resolution.

        Args:
            pkl_path (Path | str): Path to the PKL file containing the pre-recorded data.
            bin_width (float): The width of each bin in the histogram.
            resolution (tuple[int, int]): The spatial resolution of the sensor (width, height).
        """
        self._pkl_path = Path(pkl_path)
        self._data = PklWriter.load_all(self._pkl_path)
        self._data_iterator = iter(self._data)
        get_logger().info(f"Loaded {len(self._data)} entries from {self._pkl_path}.")

        self._bin_width = bin_width
        self._resolution = resolution

        self._check_data()

    def _check_data(self):
        """
        Checks the loaded data for consistency and validity, ensuring that the entries
        contain histograms with the correct resolution.
        """
        assert len(self._data) > 0, f"No data found in {self._pkl_path}"

        entry = self._data[0]
        assert "histogram" in entry, f"Entry does not contain histogram: {entry}"

        histogram = entry["histogram"]
        assert histogram[..., 0].size == np.prod(
            self._resolution
        ), f"Invalid resolution: {histogram.shape[:2]} != {self._resolution}"

    def accumulate(self, num_samples: int, *, average: bool = True) -> np.ndarray:
        """
        Accumulates the specified number of histogram samples from the pre-recorded data.

        Args:
            num_samples (int): The number of samples to accumulate.
            average (bool): Whether to average the accumulated samples. Defaults to True.

        Returns:
            np.ndarray: The accumulated histogram data, averaged if requested.
        """
        if self._data_iterator is None:
            get_logger().error("No data available.")
            return None

        histograms = []
        for _ in range(num_samples):
            try:
                entry = next(self._data_iterator)
            except StopIteration:
                get_logger().error("No more data available.")
                self._data_iterator = None
                break

            histograms.append(entry["histogram"])
        else:
            histograms = np.array(histograms)
            if average and len(histograms) > 1:
                histograms = np.mean(histograms, axis=0)

            histograms = np.squeeze(histograms)
            return histograms

        return None

    @property
    def num_bins(self) -> int:
        """
        Returns the number of bins in the sensor's histogram.

        Returns:
            int: The number of bins in the histogram.
        """
        return self._data[0]["histogram"].shape[-1]

    @property
    def bin_width(self) -> float:
        """
        Returns the width of each bin in the histogram.

        Returns:
            float: The width of each bin.
        """
        return self._bin_width

    @property
    def resolution(self) -> tuple[int, int]:
        """
        Returns the resolution of the sensor as a tuple (width, height).

        Returns:
            tuple[int, int]: The resolution (width, height) of the sensor.
        """
        return self._resolution

    @property
    def is_okay(self) -> bool:
        """
        Checks if the data iterator is still active and not exhausted.

        Returns:
            bool: True if the iterator is active, False if exhausted.
        """
        return self._data_iterator is not None

    def close(self) -> None:
        """
        Closes the sensor connection. This method is a no-op for this fake sensor.
        """
        pass
