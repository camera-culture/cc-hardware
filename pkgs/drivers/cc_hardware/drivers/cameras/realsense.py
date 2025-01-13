"""Camera driver for Intel RealSense devices.

The :class:`~cc_hardware.drivers.cameras.realsense.RealsenseCamera` class is a wrapper
around the PyRealSense library for interfacing with a D435i RealSense camera. It
provides a simple interface for capturing images and setting camera parameters. It is
implemented as a singleton to ensure that only one instance of the camera is created.
It will capture both color and depth images, but the
:func:`~cc_hardware.drivers.cameras.realsense.RealsenseCamera.accumulate` method will
only return the color image by default (set ``return_depth=True`` to return the depth
image, as well).
"""

import multiprocessing
from typing import override

import numpy as np
import pyrealsense2 as rs

from cc_hardware.drivers.cameras.camera import Camera, CameraConfig
from cc_hardware.utils import config_wrapper, get_logger


@config_wrapper
class RealsenseConfig(CameraConfig):
    """
    Configuration for Camera sensors.
    """

    camera_index: int = 0
    start_pipeline_once: bool = True
    force_autoexposure: bool = True
    exposure: int | list[int] | None = None
    align: bool = True


class RealsenseCamera(Camera[RealsenseConfig]):
    """
    Camera class for Intel RealSense devices. Captures RGB and depth images
    in a background thread and stores them in a queue.
    """

    def __init__(self, config: RealsenseConfig):
        """
        Initialize a RealsenseCamera instance.

        Args:
            config (RealsenseConfig): The configuration for the RealSense camera.
        """
        super().__init__(config)

        self.camera_index = config.camera_index
        self.start_pipeline_once = config.start_pipeline_once
        self.force_autoexposure = config.force_autoexposure
        self.align = config.align

        context = multiprocessing.get_context("spawn")
        self.queue = context.Queue(maxsize=1)
        self.stop_thread = context.Event()
        self.start_thread = context.Event()
        self.has_started = context.Event()
        self.start_capture_event = context.Event()

        # Store exposure settings
        exposure = config.exposure
        self.exposure_settings = exposure if exposure is not None else []
        # Flag to check if exposure has been initialized
        self.exposure_initialized = exposure is not None

        self._start_background_capture(context)

        if self.start_pipeline_once:
            self.start_capture_event.set()
            self.has_started.wait()

        self._initialized = True

    def _start_background_capture(self, context):
        """
        Starts the background thread to initialize the camera and capture images.
        """
        self.thread = context.Process(target=self._background_capture, daemon=True)
        self.thread.start()

    def _background_capture(self):
        """Initializes the camera, continuously captures RGB, depth images, and
        stores them in the queue."""
        get_logger().info(
            f"Starting background capture for camera index {self.camera_index}"
        )

        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()

        # Enable color, depth, and IR streams
        self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        # self.rs_config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 6)

        align = rs.align(rs.stream.color)

        while not self.stop_thread.is_set():
            # Wait until capture is started
            self.start_capture_event.wait()
            try:
                get_logger().info(
                    f"Starting pipeline for camera index {self.camera_index}"
                )
                self.pipeline.start(self.rs_config)
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
                    if self.align:
                        frames = align.process(frames)

                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    # ir_frame = frames.get_infrared_frame()

                    if not color_frame or not depth_frame:
                        continue

                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    # ir_image = np.asanyarray(ir_frame.get_data())

                    # Store tuple (color_image, depth_image, ir_image) in the queue
                    try:
                        self.queue.put((color_image, depth_image, None), block=False)
                    except multiprocessing.queues.Full:
                        continue
            except Exception as ex:
                get_logger().error(f"Camera error: {ex}")
                self.stop_thread.set()
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

    @property
    def config(self) -> RealsenseConfig:
        """
        Get the RealSense configuration object.

        Returns:
          rs.config: The RealSense configuration object.
        """
        return self._config

    def accumulate(
        self,
        num_samples: int = 1,
        *,
        return_rgb: bool = True,
        return_depth: bool = False,
        return_ir: bool = False,
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

            # Clear the queue
            while not self.queue.empty():
                self.queue.get()

        try:
            color_images = []
            depth_images = []
            ir_images = []

            while len(color_images) < num_samples:
                try:
                    item = self.queue.get()

                    color_image, depth_image, ir_image = item
                    color_images.append(color_image)
                    depth_images.append(depth_image)
                    ir_images.append(ir_image)
                except IndexError:
                    continue  # Wait for more data if queue is empty

            if num_samples == 1:
                color_images = color_images[0]
                depth_images = depth_images[0]
                ir_images = ir_images[0]

            result = []
            if return_rgb:
                result.append(np.array(color_images))
            if return_depth:
                result.append(np.array(depth_images))
            if return_ir:
                result.append(np.array(ir_images))
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
            while not self.queue.empty():
                self.queue.get()
            self.thread.join()  # Wait for the thread to finish
            self.thread = None
