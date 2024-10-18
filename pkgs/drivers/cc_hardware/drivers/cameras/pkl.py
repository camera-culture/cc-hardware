from pathlib import Path

import numpy as np

from cc_hardware.drivers.camera import Camera
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.writers import PklWriter


class PklCamera(Camera):
    def __init__(self, pkl_path: Path | str):
        self._pkl_path = Path(pkl_path)
        self._data = PklWriter.load_all(self._pkl_path)
        self._data_iterator = iter(self._data)
        get_logger().info(f"Loaded {len(self._data)} entries from {self._pkl_path}.")

        self._check_data()

    def _check_data(self):
        assert len(self._data) > 0, f"No data found in {self._pkl_path}"

        entry = self._data[0]
        assert "images" in entry, f"Entry does not contain images: {entry}"

        images = entry["images"]
        assert len(images) == 1, f"Invalid number of images: {len(images)} != 1"

    def accumulate(self, num_samples: int) -> np.ndarray:
        if self._data_iterator is None:
            get_logger().error("No data available.")
            return None

        # Get the next num_samples entries
        images = []
        for _ in range(num_samples):
            try:
                entry = next(self._data_iterator)
            except StopIteration:
                get_logger().error("No more data available.")
                self._data_iterator = None
                break

            assert (
                len(entry["images"]) == 1
            ), f"Invalid number of images: {len(entry['images'])} != 1"
            images.append(entry["images"][0])
        else:
            return np.array(images)

        return None

    @property
    def resolution(self) -> tuple[int, int]:
        return self._data[0]["image"].shape[:2]

    @property
    def distortion_coefficients(self) -> np.ndarray:
        # TODO: Load from pkl
        return np.array([-0.036, -0.145, 0.001, 0.0, 1.155])

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        # TODO: Load from pkl
        return np.array(
            [[1815.5, 0.0, 0.0], [0.0, 1817.753, 0.0], [721.299, 531.352, 1.0]]
        )

    @property
    def is_okay(self) -> bool:
        # Return whether the iterator is not exhausted
        return self._data_iterator is not None

    def close(self) -> None:
        pass
