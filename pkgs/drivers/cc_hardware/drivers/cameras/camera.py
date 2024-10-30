from abc import abstractmethod

import numpy as np

from cc_hardware.drivers.sensor import Sensor
from cc_hardware.utils.registry import Registry


class Camera(Registry, Sensor):
    """
    Abstract base class for a Camera sensor, extending the Sensor class. 
    Defines methods and properties for specific for cameras.
    """

    @abstractmethod
    def accumulate(self, num_samples: int, *, average: bool) -> np.ndarray:
        """
        Accumulate a specified number of samples from the camera.

        Args:
            num_samples (int): Number of samples to accumulate.

        Keyword Args:
            average (bool): Whether to average the accumulated samples.

        Returns:
            np.ndarray: The accumulated samples as an array.
        """
        pass

    @property
    @abstractmethod
    def distortion_coefficients(self) -> np.ndarray:
        """
        Get the camera's distortion coefficients.

        Returns:
            np.ndarray: An array representing the distortion coefficients.
        """
        pass

    @property
    @abstractmethod
    def intrinsic_matrix(self) -> np.ndarray:
        """
        Get the camera's intrinsic matrix.

        Returns:
            np.ndarray: A 3x3 matrix representing the camera intrinsics.
        """
        pass

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """
        Get the camera's resolution.

        Returns:
            tuple[int, int]: A tuple containing the width and height of 
                             the camera resolution.
        """
        pass
