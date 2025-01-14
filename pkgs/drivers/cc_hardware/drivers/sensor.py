"""Base classes for sensors and sensor data processing."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from cc_hardware.utils import Component, Config, config_wrapper
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.setting import Setting


@config_wrapper
class SensorConfig(Config):
    """Configuration for sensors.

    When defining a new sensor, create a subclass of this configuration class
    and add any necessary parameters.

    Attributes:
        instance (str): The sensor instance name. Used to instantiate the sensor from
            the configuration.
    """

    @property
    def settings(self) -> dict[str, Setting]:
        """Retrieves the sensor settings."""
        return {}


class Sensor[T: SensorConfig](Component[T]):
    """Abstract base class for sensors.

    Args:
        config (SensorConfig): The sensor configuration.
    """

    def __init__(self, config: T):
        super().__init__(config)

    @property
    def settings(self) -> dict[str, Setting]:
        """Retrieves the sensor settings."""
        return self.config.settings

    def update(self, **kwargs) -> bool:
        """
        Updates the sensor configuration with provided keyword arguments. If there are
        any changes given via the kwargs or in the settings, the configuration is sent
        to the sensor.

        Args:
            **kwargs: Configuration parameters to update. Keys must match
                the fields of SensorConfig.

        Returns:
            bool: True if the configuration was updated. False if no changes were made.
        """
        dirty = False
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                dirty = True
            else:
                get_logger().warning(f"Unknown config key: {key}")

        for name, setting in self.settings.items():
            if setting.dirty:
                dirty = True
                setattr(self.config, name, setting.value)
                setting.dirty = False

        return dirty

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
