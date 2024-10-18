from abc import ABC, abstractmethod

from cc_hardware.drivers.sensor import Sensor


class Algorithm(ABC):
    def __init__(self, sensor: Sensor):
        self._sensor = sensor

    @abstractmethod
    def run(self):
        pass

    @property
    @abstractmethod
    def is_okay(self) -> bool:
        pass

    def close(self):
        pass
