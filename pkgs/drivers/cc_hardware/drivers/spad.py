from abc import abstractmethod

from cc_hardware.drivers.sensor import Sensor


class SPADSensor(Sensor):
    @property
    @abstractmethod
    def bin_width(self) -> float:
        pass
