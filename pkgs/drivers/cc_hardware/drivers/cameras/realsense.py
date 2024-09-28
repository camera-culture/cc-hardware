import threading
from typing import override, Tuple, List, Union

import numpy as np
import pyrealsense2 as rs

from cc_hardware.drivers.camera import Camera
from cc_hardware.utils.blocking_deque import BlockingDeque
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.singleton import SingletonABCMeta


class RealsenseCamera(Camera, metaclass=SingletonABCMeta):
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.queue = BlockingDeque(maxlen=10)
        self.stop_thread = threading.Event()
        self.has_started = threading.Event()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable both color and depth streams
        # NOTE: Set frame rate to 6, otherwise there will be issues when
        # arduino's are connected in parallel (like the ardunio's serial
        # output was found to be corrupted)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
        
        # Create a point cloud object
        self.pc = rs.pointcloud()

        self._start_background_capture()
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
        try:
            self.pipeline.start(self.config)
            get_logger().info(f"Started background capture for camera index {self.camera_index}")

            self.has_started.set()

            while not self.stop_thread.is_set():
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

            self.pipeline.stop()
        except Exception as ex:
            get_logger().error(f"Camera error: {ex}")
        finally:
            get_logger().info(
                f"Stopped background capture for camera index {self.camera_index}"
            )
            self.has_started.clear()

    def accumulate(
        self, 
        num_samples: int, 
        return_rgb: bool = True, 
        return_depth: bool = True, 
        return_pc: bool = True
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
        """Accumulates RGB, depth images, and point clouds from the queue."""
        color_images = []
        depth_images = []
        point_clouds = []

        while len(color_images) < num_samples:
            if len(self.queue) >= num_samples:
                items = list(self.queue)[-num_samples:]
            else:
                items = list(self.queue)[-len(self.queue):]

            for color_image, depth_image, point_cloud in items:
                color_images.append(color_image)
                depth_images.append(depth_image)
                point_clouds.append(point_cloud)

        result = []
        if return_rgb:
            result.append(np.array(color_images))
        if return_depth:
            result.append(np.array(depth_images))
        if return_pc:
            result.append(np.array(point_clouds))
        
        return tuple(result) if len(result) > 1 else result[0]

    @property
    @override
    def resolution(self) -> Tuple[int, int]:
        """Return the resolution (width, height) of the camera."""
        return 640, 480  # Should match the resolution in the config

    @property
    @override
    def is_okay(self) -> bool:
        """Check if the camera is properly initialized."""
        return self._initialized and self.has_started.is_set()

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
        if self.thread is not None:
            self.thread.join()  # Wait for the thread to finish
            self.thread = None

        if self.has_started.is_set():
            self.pipeline.stop()
