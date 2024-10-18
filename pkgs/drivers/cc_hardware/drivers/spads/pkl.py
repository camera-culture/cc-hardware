from pathlib import Path

import numpy as np

from cc_hardware.drivers.spad import SPADSensor
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.writers import PklWriter


class PklSPADSensor(SPADSensor):
    """This is a fake SPAD sensor which loads data from a PKL file."""

    def __init__(
        self, pkl_path: Path | str, *, bin_width: float, resolution: tuple[int, int]
    ):
        self._pkl_path = Path(pkl_path)
        self._data = PklWriter.load_all(self._pkl_path)
        self._data_iterator = iter(self._data)
        get_logger().info(f"Loaded {len(self._data)} entries from {self._pkl_path}.")

        self._bin_width = bin_width
        self._resolution = resolution

        self._check_data()

    def _check_data(self):
        assert len(self._data) > 0, f"No data found in {self._pkl_path}"

        entry = self._data[0]
        assert "histogram" in entry, f"Entry does not contain histogram: {entry}"

        histogram = entry["histogram"]
        assert histogram[..., 0].size == np.prod(
            self._resolution
        ), f"Invalid resolution: {histogram.shape[:2]} != {self._resolution}"

    def accumulate(self, num_samples: int, *, average: bool = True) -> np.ndarray:
        if self._data_iterator is None:
            get_logger().error("No data available.")
            return None

        # Get the next num_samples entries
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
        return self._data[0]["histogram"].shape[-1]

    @property
    def bin_width(self) -> float:
        return self._bin_width

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def is_okay(self) -> bool:
        # Return whether the iterator is not exhausted
        return self._data_iterator is not None

    def close(self) -> None:
        pass
