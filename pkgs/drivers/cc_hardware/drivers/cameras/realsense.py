import threading
from typing import override, Tuple, List, Union

import numpy as np
import pyrealsense2 as rs

from cc_hardware.drivers.camera import Camera
from cc_hardware.utils.blocking_deque import BlockingDeque
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.singleton import SingletonABCMeta


class RealsenseCamera(Camera, metaclass=SingletonABCMeta):
    def __init__(self, camera_index: int = 0, start_pipeline_once: bool = True):
        self.camera_index = camera_index
        self.start_pipeline_once = start_pipeline_once
        self.queue = BlockingDeque(maxlen=10)
        self.stop_thread = threading.Event()
        self.has_started = threading.Event()
        self.start_capture_event = threading.Event()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable both color and depth streams
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 6)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
        
        # Create a point cloud object
        self.pc = rs.pointcloud()

        self._start_background_capture()
        
        if self.start_pipeline_once:
            self.start_capture_event.set()
            self.has_started.wait()
        
        self._initialized = True

    def _start_background_capture(self):
        """Starts the background thread to initialize the camera and capture images."""
        self.thread = threading.Thread(target=self._background_capture)
        self.thread.start()

    def _background_capture(self):
        """Initializes the camera, continuously captures RGB, depth images, and point clouds, and stores them in the queue."""
        get_logger().info(
            f"Starting background capture for camera index {self.camera_index}"
        )
        while not self.stop_thread.is_set():
            # Wait until capture is started
            self.start_capture_event.wait()
            try:
                get_logger().info(f"Starting pipeline for camera index {self.camera_index}")
                self.pipeline.start(self.config)
                get_logger().info(f"Pipeline started for camera index {self.camera_index}")

                # Run exposure initialization
                self._initialize_exposure()

                self.has_started.set()

                while not self.stop_thread.is_set() and self.start_capture_event.is_set():
                    frames = self.pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    
                    if not color_frame or not depth_frame:
                        continue
                    
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    # Calculate the point cloud
                    self.pc.map_to(color_frame)
                    point_cloud = self.pc.calculate(depth_frame)
                    points = point_cloud.get_vertices()
                    point_cloud_np = np.asanyarray(points).view(np.float32).reshape(-1, 3)

                    # Store tuple (color_image, depth_image, point_cloud) in queue
                    self.queue.append((color_image, depth_image, point_cloud_np))
            except Exception as ex:
                get_logger().error(f"Camera error: {ex}")
            finally:
                # Stop the pipeline and reset events
                get_logger().info(f"Stopping pipeline for camera index {self.camera_index}")
                self.pipeline.stop()
                self.has_started.clear()
                self.start_capture_event.clear()

        get_logger().info(
            f"Background capture thread ending for camera index {self.camera_index}"
        )

    def _initialize_exposure(self):
        """Initialize auto-exposure for all sensors and then fix the exposure settings."""
        get_logger().debug("Initializing exposure...")

        device = self.pipeline.get_active_profile().get_device()
        sensors = device.query_sensors()

        # Enable auto-exposure for a few frames to stabilize
        for _ in range(30):  # Let it run for 30 frames to stabilize the exposure
            frames = self.pipeline.wait_for_frames()
            for sensor in sensors:
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
        
        # Disable auto-exposure and lock the current exposure settings
        for sensor in sensors:
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 0)

    def accumulate(
        self, 
        num_samples: int, 
        return_rgb: bool = True, 
        return_depth: bool = True, 
        return_pc: bool = True
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
        """Accumulates RGB, depth images, and point clouds from the queue."""
        if not self.start_pipeline_once:
            self.start_capture_event.set()
            self.has_started.wait()
            self.queue.clear()

        try:
            color_images = []
            depth_images = []
            point_clouds = []

            while len(color_images) < num_samples:
                try:
                    item = self.queue.popleft()

                    color_image, depth_image, point_cloud = item
                    color_images.append(color_image)
                    depth_images.append(depth_image)
                    point_clouds.append(point_cloud)
                except IndexError:
                    continue  # Wait for more data if queue is empty

            result = []
            if return_rgb:
                result.append(np.array(color_images))
            if return_depth:
                result.append(np.array(depth_images))
            if return_pc:
                result.append(np.array(point_clouds))
            
            return tuple(result) if len(result) > 1 else result[0]
        finally:
            if not self.start_pipeline_once:
                self.start_capture_event.clear()
                self.has_started.clear()

    @property
    @override
    def resolution(self) -> Tuple[int, int]:
        """Return the resolution (width, height) of the camera."""
        return 1920, 1080  # Should match the resolution in the config

    @property
    @override
    def is_okay(self) -> bool:
        """Check if the camera is properly initialized."""
        return self._initialized and (self.has_started.is_set() or not self.start_pipeline_once)

    @property
    @override
    def intrinsic_matrix(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @override
    def distortion_coefficients(self) -> np.ndarray:
        raise NotImplementedError

    @override
    def close(self):
        """Stops the background capture thread and deinitializes the camera."""
        self.stop_thread.set()  # Signal the background thread to stop
        self.start_capture_event.set()  # Unblock the thread if waiting
        if self.thread is not None:
            self.thread.join()  # Wait for the thread to finish
            self.thread = None

        if self.has_started.is_set():
            self.pipeline.stop()
