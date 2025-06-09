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
    Camera class for Intel RealSense devices. Captures RGB and depth images on the main
    thread without using background workers.
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
        self.align_streams = config.align

        # Store exposure settings
        exposure = config.exposure
        self.exposure_settings = exposure if exposure is not None else []
        self.exposure_initialized = exposure is not None

        # RealSense pipeline setup
        self.pipeline: rs.pipeline | None = None
        self.rs_config: rs.config | None = None
        self.align: rs.align | None = None
        self._pipeline_started = False

        # Initialize pipeline configuration
        self._setup_pipeline()

        if self.start_pipeline_once:
            self._start_pipeline()

        self._initialized = True

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _setup_pipeline(self) -> None:
        """Configure the RealSense pipeline."""
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()

        # Enable color and depth streams
        self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)

        if self.align_streams:
            self.align = rs.align(rs.stream.color)

    def _start_pipeline(self) -> None:
        """Start the RealSense pipeline and initialize exposure settings."""
        if self._pipeline_started:
            return

        get_logger().info(f"Starting pipeline for camera index {self.camera_index}")
        self.pipeline.start(self.rs_config)

        device = self.pipeline.get_active_profile().get_device()
        sensors = device.query_sensors()

        if not self.exposure_initialized or self.force_autoexposure:
            self._initialize_exposure(sensors)
        else:
            if isinstance(self.exposure_settings, int):
                self.exposure_settings = [self.exposure_settings] * len(sensors)

            get_logger().debug("Re-applying exposure settings...")
            for sensor, exposure_value in zip(sensors, self.exposure_settings):
                if exposure_value is not None and sensor.supports(rs.option.exposure):
                    sensor.set_option(rs.option.exposure, exposure_value)
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 0)
            get_logger().debug("Exposure settings re-applied.")

        self._pipeline_started = True
        get_logger().info(f"Pipeline started for camera index {self.camera_index}")

    def _stop_pipeline(self) -> None:
        """Stop the RealSense pipeline."""
        if not self._pipeline_started:
            return
        get_logger().info(f"Stopping pipeline for camera index {self.camera_index}")
        self.pipeline.stop()
        self._pipeline_started = False

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

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

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
        Accumulates RGB and depth images directly from the pipeline.

        Args:
          num_samples (int): Number of image samples to accumulate.

        Keyword Args:
          return_rgb (bool): Whether to return RGB images. Defaults to True.
          return_depth (bool): Whether to return depth images. Defaults to False.

        Returns:
          List[np.ndarray] or Tuple[List[np.ndarray], List[np.ndarray]]:
            Accumulated images. Returns a list of RGB images, depth images, or both.
        """
        if not self._pipeline_started:
            self._start_pipeline()

        color_images: list[np.ndarray] = []
        depth_images: list[np.ndarray] = []
        ir_images: list[np.ndarray] = []

        try:
            while len(color_images) < num_samples:
                frames = self.pipeline.wait_for_frames()
                if self.align is not None:
                    frames = self.align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_images.append(np.asanyarray(color_frame.get_data()))
                depth_images.append(np.asanyarray(depth_frame.get_data()))
                ir_images.append(None)

            if num_samples == 1:
                color_images = color_images[0]
                depth_images = depth_images[0]
                ir_images = ir_images[0]

            result: list[np.ndarray] = []
            if return_rgb:
                result.append(np.array(color_images))
            if return_depth:
                result.append(np.array(depth_images))
            if return_ir:
                result.append(np.array(ir_images))
            return tuple(result) if len(result) > 1 else result[0]
        finally:
            if not self.start_pipeline_once:
                self._stop_pipeline()

    # --------------------------------------------------------------------- #
    # Camera interface implementation
    # --------------------------------------------------------------------- #

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
        return self._initialized and self._pipeline_started

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

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #

    @override
    def close(self) -> None:
        """
        Deinitializes the camera and stops the pipeline if running.
        """
        self._stop_pipeline()
