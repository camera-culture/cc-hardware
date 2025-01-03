"""The :class:`~cc_hardware.drivers.cameras.flir.FlirCamera` class is a wrapper around
the PySpin library for interfacing with FLIR cameras.

It provides a simple interface for capturing images and setting camera
parameters. It is implemented as a singleton to ensure that only one instance of the
camera is created. To create a new instance of the camera, one has to override the base
:class:`~cc_hardware.drivers.cameras.flir.FlirCamera` class and implement the
:func:`~drivers.cameras.flir.FlirCamera.distortion_coefficients` and
:func:`~drivers.cameras.flir.FlirCamera.intrinsic_matrix` methods.

Example:

.. code-block:: python

    class GrasshopperFlirCamera(FlirCamera):
        \"\"\"
        Specialized camera class for a Grasshopper FLIR camera model.
        Inherits from FlirCamera and provides specific intrinsic and
        distortion parameters.
        \"\"\"

        DISTORTION_COEFFICIENTS = np.array([-0.036, -0.145, 0.001, 0.0, 1.155])
        INTRINSIC_MATRIX = np.array(
            [[1815.5, 0.0, 0.0], [0.0, 1817.753, 0.0], [721.299, 531.352, 1.0]]
        )

        @property
        @override
        def distortion_coefficients(self) -> np.ndarray:
            \"\"\"
            Get the distortion coefficients of the Grasshopper FLIR camera.

            Returns:
                np.ndarray: Array of distortion coefficients.
            \"\"\"
            return self.DISTORTION_COEFFICIENTS

        @property
        @override
        def intrinsic_matrix(self) -> np.ndarray:
            \"\"\"
            Get the intrinsic matrix of the Grasshopper FLIR camera.

            Returns:
                np.ndarray: A 3x3 array representing the intrinsic matrix
                            of the camera.
            \"\"\"
            return self.INTRINSIC_MATRIX

PySpin Installation
-------------------

You will need to install PySpin and Spinnaker
`as usual <https://www.flir.co.uk/products/spinnaker-sdk>`_.
As of writing (2024-09-21), PySpin only supports <= 3.10. To install PySpin on newer
versions of Python, you can use the following steps:

.. code-block:: bash

    # After installing Spinnaker, you're instructed to run the following command:
    tar -xvzf spinnaker_python-<version>-cp<version>-<os>-<version>-<arch>.tar.gz
    pip install spinnaker_python-<version>-cp<version>-<os>-<version>-<arch>.whl

    # But this will fail for python versions > 3.10. To install on newer versions,
    # replace the cp<version> with your python version. For instance, for python 3.11 on
    # M2 Mac, the command would turn from
    tar -xvzf spinnaker_python-4.1.0.172-cp310-cp310-macosx_13_0_arm64.tar.gz
    pip instal spinnaker_python-4.1.0.172-cp310-cp310-macosx_13_0_arm64.whl
    # To
    tar -xvzf spinnaker_python-4.1.0.172-cp310-cp310-macosx_13_0_arm64.tar.gz
    mv spinnaker_python-4.1.0.172-cp310-cp310-macosx_13_0_arm64.whl \
        spinnaker_python-4.1.0.172-cp311-cp311-macosx_13_0_arm64.whl
    pip install spinnaker_python-4.1.0.172-cp311-cp311-macosx_13_0_arm64.whl

    # And then go to your site packages and do
    mv _PySpin.cpython-310-darwin.so _PySpin.cpython-311-darwin.so

.. warning::

    Installing PySpin on newer versions of Python is not officially supported and may
    cause issues. Use at your own risk.

"""

import threading
from typing import override

import numpy as np
import PySpin

from cc_hardware.drivers.cameras.camera import Camera
from cc_hardware.utils.blocking_deque import BlockingDeque
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.registry import register
from cc_hardware.utils.singleton import SingletonABCMeta


@register
class FlirCamera(Camera, metaclass=SingletonABCMeta):
    """
    A singleton camera class for FLIR cameras using the PySpin library.
    Captures images in a background thread and stores them in a queue.
    """

    def __init__(self, camera_index: int = 0):
        """
        Initialize a FlirCamera instance.

        Args:
            camera_index (int, optional): Index of the camera to initialize.
                                          Defaults to 0.
        """
        self.camera_index = camera_index
        self.queue = BlockingDeque(maxlen=10)
        self.stop_thread = threading.Event()
        self.has_started = threading.Event()
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        assert self.camera_index < self.cam_list.GetSize(), "Invalid camera index."
        assert self.cam_list.GetSize() > 0, "No cameras detected."
        self.cam = self.cam_list[self.camera_index]

        self._start_background_capture()
        self.has_started.wait()
        self._initialized = True

    def _start_background_capture(self):
        """Starts the background thread to initialize the camera and capture images."""
        self.thread = threading.Thread(target=self._background_capture)
        self.thread.start()

    def _background_capture(self):
        """
        Initializes the camera, continuously captures images, and stores
        them in the queue.
        """
        get_logger().info(
            f"Starting background capture for camera index {self.camera_index}"
        )
        try:
            self.cam.Init()
            self.cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
            self.cam.UserSetLoad()
            self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

            self.cam.BeginAcquisition()
            self.has_started.set()

            while not self.stop_thread.is_set():
                image = self._capture_image(self.cam)
                self.queue.append(image)

            self.cam.EndAcquisition()
        except PySpin.SpinnakerException as ex:
            get_logger().error(f"Camera error: {ex}")
        finally:
            self.cam.DeInit()
            del self.cam
            self.cam_list.Clear()
            self.system.ReleaseInstance()
        get_logger().info(
            f"Stopped background capture for camera index {self.camera_index}"
        )

    def accumulate(self, num_samples: int, *, average: bool = False) -> np.ndarray:
        """
        Accumulate a specified number of image samples from the queue.

        Args:
            num_samples (int): Number of image samples to accumulate.

        Keyword Args:
            average (bool, optional): Whether to average the accumulated
                                      images. Defaults to False.

        Returns:
            np.ndarray: Array containing the accumulated or averaged images.
                        Returns None if no data is available.
        """
        images = []
        while len(images) < num_samples:
            if len(self.queue) >= num_samples:
                images.extend(list(self.queue)[-num_samples:])
            else:
                images.extend(list(self.queue)[-len(self.queue) :])

        if average and len(images) > 1:
            return np.mean(images, axis=0).astype(dtype=images[0].dtype)
        return np.array(images)

    def _capture_image(self, cam):
        """
        Capture a single image from the camera.

        Args:
            cam: The camera instance to capture the image from.

        Returns:
            np.ndarray: The captured image as a numpy array.
        """
        image_result = cam.GetNextImage()
        assert (
            not image_result.IsIncomplete()
        ), f"Image incomplete with status: {image_result.GetImageStatus()}"
        image_data = np.copy(image_result.GetNDArray())
        image_result.Release()
        return image_data

    @property
    @override
    def resolution(self) -> tuple[int, int]:
        """
        Get the resolution (width, height) of the camera.

        Returns:
            tuple[int, int]: A tuple containing the width and height
                             of the camera.
        """
        return int(self.cam.Width.GetValue()), int(self.cam.Height.GetValue())

    @property
    @override
    def is_okay(self) -> bool:
        """
        Check if the camera is properly initialized.

        Returns:
            bool: True if the camera is initialized and streaming properly,
                  False otherwise.
        """
        if not hasattr(self, "cam"):
            return False

        is_initialized = self.cam.IsInitialized()
        is_streaming = self.cam.IsStreaming()
        has_started = self.has_started.is_set()

        return is_initialized and (not has_started or is_streaming)

    @override
    def close(self):
        """Stops the background capture thread and deinitializes the camera."""
        self.stop_thread.set()  # Signal the background thread to stop
        if self.thread is not None:
            self.thread.join()  # Wait for the thread to finish
            self.thread = None

        if hasattr(self, "cam") and self.cam is not None:
            if self.cam.IsStreaming():
                self.cam.EndAcquisition()

            if self.cam.IsInitialized():
                self.cam.DeInit()

        if hasattr(self, "system") and self.system is not None:
            self.cam_list.Clear()
            self.system.ReleaseInstance()


@register
class GrasshopperFlirCamera(FlirCamera):
    """
    Specialized camera class for a Grasshopper FLIR camera model.
    Inherits from FlirCamera and provides specific intrinsic and
    distortion parameters.
    """

    DISTORTION_COEFFICIENTS = np.array([-0.036, -0.145, 0.001, 0.0, 1.155])
    INTRINSIC_MATRIX = np.array(
        [[1815.5, 0.0, 0.0], [0.0, 1817.753, 0.0], [721.299, 531.352, 1.0]]
    )

    @property
    @override
    def distortion_coefficients(self) -> np.ndarray:
        """
        Get the distortion coefficients of the Grasshopper FLIR camera.

        Returns:
            np.ndarray: Array of distortion coefficients.
        """
        return self.DISTORTION_COEFFICIENTS

    @property
    @override
    def intrinsic_matrix(self) -> np.ndarray:
        """
        Get the intrinsic matrix of the Grasshopper FLIR camera.

        Returns:
            np.ndarray: A 3x3 array representing the intrinsic matrix
                        of the camera.
        """
        return self.INTRINSIC_MATRIX
