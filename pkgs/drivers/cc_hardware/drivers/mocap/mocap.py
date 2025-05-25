from abc import abstractmethod

from cc_hardware.drivers.sensor import Sensor, SensorConfig
from cc_hardware.utils import config_wrapper


@config_wrapper
class MotionCaptureSensorConfig(SensorConfig):
    """Configuration for SPAD sensors."""

    pass


class MotionCaptureSensor[T: MotionCaptureSensorConfig](Sensor[T]):
    """
    An abstract base class for motion capture sensors.
    """

    @abstractmethod
    def accumulate(self, num_samples: int = 1):
        """
        Accumulates the specified number of pose samples from the sensor.

        Args:
            num_samples (int): The number of samples to accumulate into the pose.
                The accumulation method (i.e. summing, averaging) may vary depending on
                the sensor. Defaults to 1.
        """
        pass
