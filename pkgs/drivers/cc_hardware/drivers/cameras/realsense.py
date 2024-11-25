"""Camera driver for Intel RealSense devices."""

import threading
from typing import override

import numpy as np
import pyrealsense2 as rs

from cc_hardware.drivers.cameras.camera import Camera
from cc_hardware.utils.blocking_deque import BlockingDeque
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.registry import register
from cc_hardware.utils.singleton import SingletonABCMeta


@register
class RealsenseCamera(Camera, metaclass=SingletonABCMeta):
    """
    Camera class for Intel RealSense devices. Captures RGB and depth images
    in a background thread and stores them in a queue.
    """

    def __init__(
        self,
        camera_index: int = 0,
        start_pipeline_once: bool = True,
        force_autoexposure: bool = False,
        exposure: int | list[int] | None = None,
    ):
        """
        Initialize a RealsenseCamera instance.

        Args:
          camera_index (int): Index of the camera to initialize. Defaults to 0.
          start_pipeline_once (bool): Whether to start the pipeline only once.
            Defaults to True.
          force_autoexposure (bool): Whether to force autoexposure initialization.
            Defaults to False.
        """
        self.camera_index = camera_index
        self.start_pipeline_once = start_pipeline_once
        self.force_autoexposure = force_autoexposure

        self.queue = BlockingDeque(maxlen=10)
        self.stop_thread = threading.Event()
        self.has_started = threading.Event()
        self.start_capture_event = threading.Event()
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable both color and depth streams
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 6)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)

        # Store exposure settings
        self.exposure_settings = exposure if exposure is not None else []
        # Flag to check if exposure has been initialized
        self.exposure_initialized = exposure is not None

        self._start_background_capture()

        if self.start_pipeline_once:
            self.start_capture_event.set()
            self.has_started.wait()

        self._initialized = True

    def _start_background_capture(self):
        """
        Starts the background thread to initialize the camera and capture images.
        """
        self.thread = threading.Thread(target=self._background_capture)
        self.thread.start()

    def _background_capture(self):
        """Initializes the camera, continuously captures RGB, depth images, and
        stores them in the queue."""
        get_logger().info(
            f"Starting background capture for camera index {self.camera_index}"
        )
        while not self.stop_thread.is_set():
            # Wait until capture is started
            self.start_capture_event.wait()
            try:
                get_logger().info(
                    f"Starting pipeline for camera index {self.camera_index}"
                )
                self.pipeline.start(self.config)
                get_logger().info(
                    f"Pipeline started for camera index {self.camera_index}"
                )

                device = self.pipeline.get_active_profile().get_device()
                sensors = device.query_sensors()

                if not self.exposure_initialized or self.force_autoexposure:
                    # Run exposure initialization
                    self._initialize_exposure(sensors)
                else:
                    if isinstance(self.exposure_settings, int):
                        self.exposure_settings = [self.exposure_settings] * len(sensors)

                    # Re-apply saved exposure settings
                    get_logger().debug("Re-applying exposure settings...")
                    for sensor, exposure_value in zip(sensors, self.exposure_settings):
                        if exposure_value is not None and sensor.supports(
                            rs.option.exposure
                        ):
                            sensor.set_option(rs.option.exposure, exposure_value)
                        if sensor.supports(rs.option.enable_auto_exposure):
                            sensor.set_option(rs.option.enable_auto_exposure, 0)
                    get_logger().debug("Exposure settings re-applied.")

                self.has_started.set()

                while (
                    not self.stop_thread.is_set() and self.start_capture_event.is_set()
                ):
                    frames = self.pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                    if not color_frame or not depth_frame:
                        continue

                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())

                    # Store tuple (color_image, depth_image) in queue
                    self.queue.append((color_image, depth_image))
            except Exception as ex:
                get_logger().error(f"Camera error: {ex}")
            finally:
                # Stop the pipeline and reset events
                get_logger().info(
                    f"Stopping pipeline for camera index {self.camera_index}"
                )
                self.pipeline.stop()
                self.has_started.clear()
                self.start_capture_event.clear()

        get_logger().info(
            f"Background capture thread ending for camera index {self.camera_index}"
        )

    def _initialize_exposure(self, sensors) -> None:
        """
        Initialize auto-exposure for all sensors and then fix the exposure settings.

        Args:
          sensors: List of sensors from the device to initialize exposure for.
        """
        get_logger().info("Initializing exposure...")

        # Enable auto-exposure for a few frames to stabilize
        get_logger().debug("Starting autoexposure procedure...")
        for _ in range(10):  # Let it run for 10 frames to stabilize the exposure
            _ = self.pipeline.wait_for_frames()
            for sensor in sensors:
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
        get_logger().debug("Finished with autoexposure procedure.")

        # Disable auto-exposure and lock the current exposure settings
        get_logger().debug("Disabling autoexposure and saving exposure settings...")
        self.exposure_settings = []
        for sensor in sensors:
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 0)
            exposure_value = (
                sensor.get_option(rs.option.exposure)
                if sensor.supports(rs.option.exposure)
                else None
            )
            self.exposure_settings.append(exposure_value)
        get_logger().debug(f"Saved exposure settings: {self.exposure_settings}")
        get_logger().debug("Disabled autoexposure.")

        self.exposure_initialized = True

    def accumulate(
        self,
        num_samples: int,
        return_rgb: bool = True,
        return_depth: bool = False,
    ) -> list[np.ndarray] | tuple[list[np.ndarray] | list[np.ndarray]]:
        """
        Accumulates RGB and depth images from the queue.

        Args:
          num_samples (int): Number of image samples to accumulate.

        Keyword Args:
          return_rgb (bool): Whether to return RGB images. Defaults to True.
          return_depth (bool): Whether to return depth images. Defaults to False.

        Returns:
          List[np.ndarray] or Tuple[List[np.ndarray], List[np.ndarray]]:
            Accumulated images. Returns a list of RGB images, depth images, or both.
        """
        if not self.start_pipeline_once:
            self.start_capture_event.set()
            self.has_started.wait()
            self.queue.clear()

        try:
            color_images = []
            depth_images = []

            while len(color_images) < num_samples:
                try:
                    item = self.queue.popleft()

                    color_image, depth_image = item
                    color_images.append(color_image)
                    depth_images.append(depth_image)
                except IndexError:
                    continue  # Wait for more data if queue is empty

            if num_samples == 1:
                color_images = color_images[0]
                depth_images = depth_images[0]

            result = []
            if return_rgb:
                result.append(np.array(color_images))
            if return_depth:
                result.append(np.array(depth_images))
            return tuple(result) if len(result) > 1 else result[0]
        finally:
            if not self.start_pipeline_once:
                self.start_capture_event.clear()
                self.has_started.clear()

    @property
    @override
    def resolution(self) -> tuple[int, int]:
        """
        Return the resolution (width, height) of the camera.

        Returns:
          Tuple[int, int]: The resolution of the color stream.
        """
        return 1920, 1080  # TODO: Should match the resolution in the config

    @property
    @override
    def is_okay(self) -> bool:
        """
        Check if the camera is properly initialized.

        Returns:
          bool: True if the camera is initialized and ready, False otherwise.
        """
        return self._initialized and (
            self.has_started.is_set() or not self.start_pipeline_once
        )

    @property
    @override
    def intrinsic_matrix(self) -> np.ndarray:
        """
        Get the intrinsic matrix of the camera.

        Returns:
          np.ndarray: The intrinsic matrix of the camera.

        Raises:
          NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError

    @property
    @override
    def distortion_coefficients(self) -> np.ndarray:
        """
        Get the distortion coefficients of the camera.

        Returns:
          np.ndarray: The distortion coefficients of the camera.

        Raises:
          NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError

    @override
    def close(self):
        """
        Stops the background capture thread and deinitializes the camera.
        """
        self.stop_thread.set()  # Signal the background thread to stop
        self.start_capture_event.set()  # Unblock the thread if waiting
        if self.thread is not None:
            self.thread.join()  # Wait for the thread to finish
            self.thread = None

        if self.has_started.is_set():
            self.pipeline.stop()
