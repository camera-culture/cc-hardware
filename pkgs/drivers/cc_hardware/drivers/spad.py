from abc import abstractmethod

from cc_hardware.drivers.sensor import Sensor


class SPADSensor(Sensor):
    @abstractmethod
    def accumulate(self, num_samples: int):
        pass

    @property
    @abstractmethod
    def num_bins(self) -> int:
        pass

    @property
    @abstractmethod
    def bin_width(self) -> float:
        pass

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        pass
