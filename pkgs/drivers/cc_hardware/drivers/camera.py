from abc import abstractmethod

import numpy as np

from cc_hardware.drivers.sensor import Sensor


class Camera(Sensor):
    @abstractmethod
    def accumulate(self, num_samples: int, *, average: bool) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def distortion_coefficients(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def intrinsic_matrix(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        pass
