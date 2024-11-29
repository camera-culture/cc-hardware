"""Base class for Single-Photon Avalanche Diode (SPAD) sensors."""

from abc import abstractmethod

from cc_hardware.drivers.sensor import Sensor

# ================


class SPADSensor(Sensor):
    """
    An abstract base class for Single-Photon Avalanche Diode (SPAD) sensors, designed
    to manage histogram-based measurements. This class defines methods and properties
    related to collecting and analyzing histogram data.

    Inherits:
        Sensor: The base class for all sensors in the system.
    """

    @abstractmethod
    def accumulate(self, num_samples: int):
        """
        Accumulates the specified number of histogram samples from the sensor.

        Args:
            num_samples (int): The number of samples to accumulate into the histogram.
                The accumulation method (i.e. summing, averaging) may vary depending on
                the sensor.
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

    @property
    @abstractmethod
    def bin_width(self) -> float:
        """
        Returns the width of each bin in the histogram. This indicates the time or
        distance resolution of the sensor, representing the range covered by each bin.
        The exact units should be documented in the derived sensor class.

        Returns:
            float: The width of each bin in the histogram.
        """
        pass

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
