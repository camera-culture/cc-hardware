import threading
from typing import override

import numpy as np
import PySpin

from cc_hardware.drivers.camera import Camera
from cc_hardware.utils.blocking_deque import BlockingDeque
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.singleton import SingletonABCMeta


class FlirCamera(Camera, metaclass=SingletonABCMeta):
    def __init__(self, camera_index: int = 0):
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
        """Initializes the camera, continuously captures images, and stores
        them in the queue."""
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
        """Accumulates images from the queue."""
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
        """Captures a single image from the camera."""
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
        """Return the resolution (width, height) of the camera."""
        return int(self.cam.Width.GetValue()), int(self.cam.Height.GetValue())

    @property
    @override
    def is_okay(self) -> bool:
        """Check if the camera is properly initialized."""
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


class GrasshopperFlirCamera(FlirCamera):
    DISTORTION_COEFFICIENTS = np.array([-0.036, -0.145, 0.001, 0.0, 1.155])
    INTRINSIC_MATRIX = np.array(
        [[1815.5, 0.0, 0.0], [0.0, 1817.753, 0.0], [721.299, 531.352, 1.0]]
    )

    @property
    @override
    def distortion_coefficients(self) -> np.ndarray:
        return self.DISTORTION_COEFFICIENTS

    @property
    @override
    def intrinsic_matrix(self) -> np.ndarray:
        return self.INTRINSIC_MATRIX
