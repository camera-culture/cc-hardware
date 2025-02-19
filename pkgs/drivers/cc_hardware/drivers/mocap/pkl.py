from typing import Any
from pathlib import Path

from cc_hardware.drivers.mocap import MotionCaptureSensor, MotionCaptureSensorConfig
from cc_hardware.utils import config_wrapper, get_logger
from cc_hardware.utils.file_handlers import PklReader

# ===============


@config_wrapper
class PklMotionCaptureSensorConfig(MotionCaptureSensorConfig):
    """Config for the pkl-based motion capture sensor."""
    pkl_path: Path | str
    key: str | None = "data"
    loop: bool = False

# ===============


class PklMotionCaptureSensor(MotionCaptureSensor[PklMotionCaptureSensorConfig]):
    """"""

    def __init__(self, config: PklMotionCaptureSensorConfig):
        super().__init__(config)

        config.pkl_path = Path(config.pkl_path)
        assert config.pkl_path.exists(), f"PKL file {config.pkl_path} does not exist."
        self._handler = PklReader(config.pkl_path)
        self._index = 0

        entry = self._handler.load(self._index)
        assert config.key in entry, f"Key '{config.key}' not found in data."

    def accumulate(self, num_samples: int = 1) -> list[Any]:
        if self._index >= len(self._handler):
            get_logger().error("No more data available.")
            return None

        data = []
        for _ in range(num_samples):
            try:
                entry = self._handler.load(self._index)
                self._index += 1
                if self._index + 1 > len(self._handler) and self.config.loop:
                    get_logger().info("Looping!")
                    self._index = 0
            except StopIteration:
                get_logger().error("No more data available.")
                break

            data.append(entry[self.config.key])

        return data[0] if len(data) == 1 else data

    @property
    def is_okay(self) -> bool:
        return len(self._handler) > self._index

    def close(self) -> None:
        """
        Closes the sensor connection. This method is a no-op for this fake sensor.
        """
        pass

