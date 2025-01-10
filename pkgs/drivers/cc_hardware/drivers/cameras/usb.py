import threading
from typing import override

import cv2
import numpy as np

from cc_hardware.drivers.cameras.camera import Camera, CameraConfig
from cc_hardware.utils import config_wrapper, get_logger, register
from cc_hardware.utils.blocking_deque import BlockingDeque
from cc_hardware.utils.singleton import SingletonABCMeta


@register
@config_wrapper
class USBCameraConfig(CameraConfig):
    """
    Configuration for generic USB Camera.

    Attributes:
        instance (str): The name/identifier for this camera instance.
        camera_index (int): The index of the camera to use (e.g., 0, 1, ...).
        start_capture_once (bool): Whether to start capturing once and keep it running
            until closed. If False, capturing starts/stops each time you call
            :func:`accumulate`.
        exposure (int | None): Desired exposure value. Note that not all USB cameras
            allow setting exposure directly, and this can vary by platform.
        force_autoexposure (bool): If True, attempt to enable auto-exposure. Not all
            cameras support this.
    """

    instance: str = "USBCamera"

    camera_index: int = 0
    start_capture_once: bool = True
    exposure: int | None = None
    force_autoexposure: bool = False


@register
class USBCamera(Camera, metaclass=SingletonABCMeta):
    """
    Camera class for a generic USB camera. Captures RGB frames in a background thread
    and stores them in a queue.
    """

    def __init__(self, config: USBCameraConfig):
        """
        Initialize a USBCamera instance.

        Args:
            config (USBCameraConfig): Configuration parameters for the USB camera.
        """
        super().__init__(config)

        self.camera_index = config.camera_index
        self.start_capture_once = config.start_capture_once
        self.exposure = config.exposure
        self.force_autoexposure = config.force_autoexposure

        self.queue = BlockingDeque(maxlen=10)
        self.stop_thread = threading.Event()
        self.has_started = threading.Event()
        self.start_capture_event = threading.Event()

        self._capture = None
        self._thread = None
        self._initialized = False

        # Start the background thread, but we won't actually open the camera
        # until we set `start_capture_event`.
        self._start_background_capture()

        # If we want to keep capturing from the start:
        if self.start_capture_once:
            self.start_capture_event.set()
            self.has_started.wait(
                timeout=5.0
            )  # Wait up to 5 seconds for initialization

        self._initialized = True

    def _start_background_capture(self):
        """
        Starts the background thread that opens the camera and reads frames 
        continuously.
        """
        self._thread = threading.Thread(target=self._background_capture, daemon=True)
        self._thread.start()

    def _background_capture(self):
        """
        Background worker thread that manages camera capture and frame retrieval.
        """
        logger = get_logger()
        logger.info(
            f"Starting background capture thread for camera index {self.camera_index}"
        )

        while not self.stop_thread.is_set():
            # Wait until capture is requested
            self.start_capture_event.wait()
            logger.info(f"Opening camera index {self.camera_index}")

            try:
                # Attempt to open the camera
                self._capture = cv2.VideoCapture(self.camera_index)
            except Exception as ex:
                logger.error(f"Error opening camera {self.camera_index}: {ex}")
                self.has_started.clear()
                self.start_capture_event.clear()
                continue

            # Set exposure settings if requested (this may not work on all 
            # platforms/cameras)
            if self._capture.isOpened():
                if self.exposure is not None:
                    # Some cameras might require a negative or different value for 
                    # manual exposure
                    # or might simply ignore this if autoexposure is forced.
                    self._capture.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure))

                if self.force_autoexposure:
                    # On some systems, enabling autoexposure might require setting 
                    # exposure to -1
                    # and enabling an autoexposure property or similar. This will vary 
                    # by camera.
                    self._capture.set(cv2.CAP_PROP_EXPOSURE, -1)

                logger.info(f"Camera {self.camera_index} opened successfully.")
                self.has_started.set()
            else:
                logger.error(f"Failed to open camera {self.camera_index}.")
                self.has_started.clear()
                self.start_capture_event.clear()
                continue

            # Read frames in a loop
            while not self.stop_thread.is_set() and self.start_capture_event.is_set():
                ret, frame = self._capture.read()
                if not ret or frame is None:
                    logger.warning(
                        f"Failed to read frame from camera {self.camera_index}."
                    )
                    continue
                # Append the frame to the queue
                self.queue.append(frame)

            # We stop capturing, release the camera
            if self._capture:
                self._capture.release()
            self.has_started.clear()
            self.start_capture_event.clear()

        logger.info(
            f"Background capture thread ending for camera index {self.camera_index}"
        )

    @property
    def config(self) -> USBCameraConfig:
        """
        USBCameraConfig: The configuration object for the USBCamera instance.
        """
        return self._config

    def accumulate(self, num_samples: int = 1) -> list[np.ndarray] | np.ndarray:
        """
        Accumulates RGB frames from the camera queue.

        Args:
            num_samples (int): Number of frames to retrieve.

        Keyword Args:
            return_rgb (bool): Whether to return RGB frames. Defaults to True.
            return_depth (bool): This parameter is ignored for a USB camera,
                as we don't capture depth frames. Defaults to False.

        Returns:
            List[np.ndarray] or np.ndarray:
                A list of frames if num_samples > 1, or a single frame if 
                num_samples == 1.
        """
        # If we're not meant to continuously capture, we start capturing now
        if not self.start_capture_once:
            self.start_capture_event.set()
            self.has_started.wait(timeout=5.0)
            self.queue.clear()

        frames = []
        try:
            while len(frames) < num_samples:
                try:
                    frame = self.queue.popleft()
                    frames.append(frame)
                except IndexError:
                    continue  # Wait for more data if the queue is empty

            # If only one sample requested, return just that single frame
            if num_samples == 1:
                return frames[0]

            return frames

        finally:
            # If we're only capturing on demand, stop immediately after accumulation
            if not self.start_capture_once:
                self.start_capture_event.clear()
                self.has_started.clear()

    @property
    @override
    def resolution(self) -> tuple[int, int]:
        """
        Return the resolution (width, height) of the camera.

        Returns:
            Tuple[int, int]: The current resolution of the camera if known,
                             otherwise a default or fallback value.
        """
        # Attempt to retrieve from an open capture if available
        if self._capture and self._capture.isOpened():
            width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)

        # Fallback values
        return (640, 480)

    @property
    @override
    def is_okay(self) -> bool:
        """
        Check if the camera is properly initialized and capturing.

        Returns:
            bool: True if the camera is initialized and capturing, False otherwise.
        """
        return self._initialized and (
            self.has_started.is_set() or not self.start_capture_once
        )

    @property
    @override
    def intrinsic_matrix(self) -> np.ndarray:
        """
        Get the intrinsic matrix of the camera.

        Returns:
            np.ndarray: The intrinsic matrix of the camera.

        Raises:
            NotImplementedError: This method is not yet implemented for a generic USB 
                camera.
        """
        raise NotImplementedError(
            "Intrinsic matrix retrieval is not implemented for USBCamera."
        )

    @property
    @override
    def distortion_coefficients(self) -> np.ndarray:
        """
        Get the distortion coefficients of the camera.

        Returns:
            np.ndarray: The distortion coefficients of the camera.

        Raises:
            NotImplementedError: This method is not yet implemented for a generic USB 
                camera.
        """
        raise NotImplementedError(
            "Distortion coefficients retrieval is not implemented for USBCamera."
        )

    @override
    def close(self):
        """
        Stops the background capture thread and releases the camera resource.
        """
        self.stop_thread.set()  # Signal the background thread to stop
        self.start_capture_event.set()  # Unblock the thread if waiting
        if self._thread is not None:
            self._thread.join()  # Wait for the thread to finish
            self._thread = None

        # Release capture if still open
        if self._capture and self._capture.isOpened():
            self._capture.release()
        self._capture = None
