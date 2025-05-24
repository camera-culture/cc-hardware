"""Base class for Single-Photon Avalanche Diode (SPAD) sensors."""

from abc import abstractmethod

from cc_hardware.utils import Component, Config, config_wrapper
from cc_hardware.drivers.sensor import Sensor, SensorConfig
from cc_hardware.utils import config_wrapper, get_logger

# ================


@config_wrapper
class SPADSensorConfig(SensorConfig):
    """Configuration for SPAD sensors."""

    pass


class SPADSensor[T: SPADSensorConfig](Sensor[T]):
    """
    An abstract base class for Single-Photon Avalanche Diode (SPAD) sensors, designed
    to manage histogram-based measurements. This class defines methods and properties
    related to collecting and analyzing histogram data.

    Inherits:
        Sensor: The base class for all sensors in the system.
    """

    @abstractmethod
    def accumulate(self, num_samples: int = 1):
        """
        Accumulates the specified number of histogram samples from the sensor.

        Args:
            num_samples (int): The number of samples to accumulate into the histogram.
                The accumulation method (i.e. summing, averaging) may vary depending on
                the sensor. Defaults to 1.
        """
        pass

    @property
    @abstractmethod
    def num_bins(self) -> int:
        """
        Returns the number of bins in the sensor's histogram. This indicates the
        number of discrete values or ranges that the sensor can measure. The total
        distance a sensor can measure is equal to the number of bins multiplied by
        the bin width.

        Returns:
            int: The total number of bins in the histogram.
        """
        pass

    @num_bins.setter
    def num_bins(self, value: int):
        """
        Sets the number of bins in the sensor's histogram. This method allows the
        number of bins to be dynamically adjusted to match the sensor's configuration.

        Args:
            value (int): The new number of bins in the histogram.
        """
        get_logger().warning(f"Setting the number of bins is not supported for {self}.")

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """
        Returns the resolution of the sensor as a tuple (width, height). This indicates
        the spatial resolution of the sensor, where the width and height refer to the
        number of pixels or sampling points in the respective dimensions.

        Returns:
            tuple[int, int]: A tuple representing the sensor's resolution
                (width, height).
        """
        pass

    @resolution.setter
    def resolution(self, value: tuple[int, int]):
        """
        Sets the resolution of the sensor. This method allows the resolution to be
        dynamically adjusted to match the sensor's configuration.

        Args:
            value (tuple[int, int]): The new resolution of the sensor as a tuple
                (width, height).
        """
        get_logger().warning(f"Setting the resolution is not supported for {self}.")
