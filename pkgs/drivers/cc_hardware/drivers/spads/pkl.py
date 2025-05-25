"""SPAD sensor driver that loads pre-recorded data from a PKL file."""

from pathlib import Path

import numpy as np

from cc_hardware.drivers.spads.spad import SPADDataType, SPADSensor, SPADSensorConfig
from cc_hardware.utils import config_wrapper, get_logger
from cc_hardware.utils.file_handlers import PklReader

# ==================


@config_wrapper
class PklSPADSensorConfig(SPADSensorConfig):
    """
    Configuration for the PklSPADSensor.

    Attributes:
        pkl_path (Path | str): Path to the PKL file containing pre-recorded data.
        index (int): Initial index for loading data from the PKL file. Default is 0.

        loop (bool): Whether to loop through the data when it reaches the end.
            Default is False.
    """

    pkl_path: Path | str
    index: int = 0

    loop: bool = False


class PklSPADSensor[T: PklSPADSensorConfig](SPADSensor[T]):
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

    def __init__(self, config: PklSPADSensorConfig):
        super().__init__(config)

        config.pkl_path = Path(config.pkl_path)
        assert config.pkl_path.exists(), f"PKL file {config.pkl_path} does not exist."
        self._handler = PklReader(config.pkl_path)
        self._num_entries = len(self._handler)
        assert self._num_entries > 0, "No data found in PKL file."
        self._index = config.index

        first_entry: dict = self._handler.load(0)
        if self.config.height is None:
            assert "height" in first_entry, "Height not found in first entry."
            self.config.height = first_entry["height"]
        if self.config.width is None:
            assert "width" in first_entry, "Width not found in first entry."
            self.config.width = first_entry["width"]
        if self.config.num_bins is None:
            assert "num_bins" in first_entry, "Number of bins not found in first entry."
            self.config.num_bins = first_entry["num_bins"]
        if self.config.fovx is None:
            assert "fovx" in first_entry, "FOV X not found in first entry."
            self.config.fovx = first_entry["fovx"]
        if self.config.fovy is None:
            assert "fovy" in first_entry, "FOV Y not found in first entry."
            self.config.fovy = first_entry["fovy"]
        if self.config.timing_resolution is None:
            assert (
                "timing_resolution" in first_entry
            ), "Timing resolution not found in first entry."
            self.config.timing_resolution = first_entry["timing_resolution"]

    @property
    def handler(self) -> PklReader:
        return self._handler

    def accumulate(
        self, num_samples: int = 1, *, index: int | None = None
    ) -> list[dict[SPADDataType, np.ndarray]] | dict[SPADDataType, np.ndarray]:
        """
        Accumulates the specified number of histogram samples from the pre-recorded
        data.

        Args:
            num_samples (int): The number of samples to accumulate.
        """
        if index is not None:
            self._index = index

        if self.config.loop:
            self._index = self._index % self._num_entries
        if self._index >= self._num_entries:
            get_logger().error("No more data available.")
            return None

        samples = []
        for _ in range(num_samples):
            try:
                entry = self._handler.load(self._index)
                self._index += 1
            except StopIteration:
                get_logger().error("No more data available.")
                break

            entry = {
                k: v
                for k, v in entry.items()
                if isinstance(k, SPADDataType) and k in self.config.data_type
            }
            samples.append(entry)

        return samples[0] if len(samples) == 1 else samples

    @property
    def is_okay(self) -> bool:
        return self._num_entries > self._index or self.config.loop

    def close(self) -> None:
        """
        Closes the sensor connection. This method is a no-op for this fake sensor.
        """
        pass
