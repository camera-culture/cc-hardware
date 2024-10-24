import threading
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from cc_hardware.utils.logger import get_logger


class Sensor(ABC):
    """Abstract base class for sensors."""

    @property
    @abstractmethod
    def is_okay(self) -> bool:
        """Checks if the sensor is operational."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the sensor and releases any resources."""
        pass

    def calibrate(self) -> bool:
        """Calibrates the sensor."""
        raise NotImplementedError("Calibration is not supported for this sensor.")

    def __del__(self):
        """Destructor to ensure the sensor is properly closed."""
        try:
            self.close()
        except Exception:
            get_logger().exception(f"Failed to close {self.__class__.__name__}.")


class SensorData(ABC):
    """Abstract base class for handling sensor data."""

    def __init__(self):
        self._data: np.ndarray = None
        self._has_data = False

    @abstractmethod
    def reset(self) -> None:
        """Resets the sensor data to its initial state."""
        pass

    @abstractmethod
    def process(self, row: list[Any]) -> None:
        """Processes a new row of data.

        Args:
          row (list[Any]): Row of sensor data to process.
        """
        pass

    @abstractmethod
    def get_data(self) -> np.ndarray:
        """Retrieves the processed sensor data.

        Returns:
          np.ndarray: The processed data.
        """
        pass


class SensorDataThreaded(SensorData):
    """Handles sensor data processing with threading support."""

    def __init__(self):
        self._ready_event = threading.Event()

    def reset(self) -> None:
        """Resets the threading event and clears any existing data."""
        self._ready_event.clear()

    def get_data(self) -> np.ndarray:
        """Waits until data is ready and retrieves it.

        Returns:
          np.ndarray: The processed data.
        """
        self._ready_event.wait()
        self._ready_event.clear()
        return self._data
