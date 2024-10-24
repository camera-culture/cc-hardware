from pathlib import Path

import cv2
import numpy as np

from cc_hardware.algos.algo import Algorithm
from cc_hardware.drivers.camera import Camera
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.writers import VideoWriter


class ArucoLocalizationAlgorithm(Algorithm):
    """An algorithm to localize ArUco markers using camera images.

    This class processes images from a camera sensor to detect ArUco markers and
    compute their poses relative to an origin marker.
    """

    def __init__(
        self,
        sensor: Camera,
        *,
        aruco_dict: int,
        marker_size: float,
        origin_id: int = -1,
        num_samples: int = 1,
        **marker_ids,
    ):
        """Initializes the ArucoLocalizationAlgorithm with specified parameters.

        Args:
            sensor (Camera): The camera sensor to use for capturing images.
            aruco_dict (int): The predefined dictionary of ArUco markers to use.
            marker_size (float): The size of the ArUco markers in meters.
            origin_id (int, optional): The ID of the origin marker. Defaults to -1.
            num_samples (int, optional): The number of samples to average over.
                Defaults to 1.

        Keyword Args:
            **marker_ids: Additional marker IDs to track, passed as keyword arguments.
        """
        self._sensor: Camera = sensor

        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        aruco_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        self._marker_size = marker_size
        self._origin_id = origin_id
        self._num_samples = num_samples
        self._marker_ids = marker_ids  # Store additional marker IDs as a dict

        self._is_okay = True

    def run(
        self,
        *,
        show: bool = False,
        save: bool = False,
        filename: Path | str | None = None,
        return_images: bool = False,
    ):
        """Processes images and returns the localization results.

        Args:
            show (bool, optional): Whether to display the image with detected markers.
                Defaults to False.
            save (bool, optional): Whether to save the image with detected markers.
                Defaults to False.
            filename (Path | str | None, optional): The filename to save the image or
                video. Defaults to None.
            return_images (bool, optional): Whether to return the processed images.
                Defaults to False.

        Returns:
            dict: A dictionary containing localization results for the specified markers.
            list: A list of processed images (if return_images is True).
        """
        results = []
        for _ in range(self._num_samples):
            results.append(self._process_image(show=show, save=save, filename=filename))

        # Get the images
        images = [r.pop("image") for r in results if "image" in r]

        # Average the results
        result = {}
        keys = set([key for r in results for key in r.keys()])
        for key in keys:
            if all(key in r for r in results):
                result[key] = np.median([r[key] for r in results], axis=0)

        results = {key: result.get(key) for key in self._marker_ids}
        if return_images:
            return results, images
        return results

    def _process_image(
        self,
        *,
        show: bool = False,
        save: bool = False,
        filename: Path | str | None = None,
    ) -> dict:
        """Processes a single image to compute poses of detected markers.

        Args:
            show (bool, optional): Whether to display the image with detected markers.
                Defaults to False.
            save (bool, optional): Whether to save the image with detected markers.
                Defaults to False.
            filename (Path | str | None, optional): The filename to save the image or
                video. Defaults to None.

        Returns:
            dict: A dictionary containing poses of detected markers and the image.
        """
        image = self._sensor.accumulate(1)
        if image is None:
            get_logger().error("No image available.")
            return {}
        image = np.squeeze(image)

        # Detect markers
        corners, ids, _ = self._detector.detectMarkers(image)
        if ids is None:
            get_logger().warning("No markers detected.")
            return {}

        # Show/save the results
        if show or save:
            vis_image = image.copy()
            if len(image.shape) == 2:
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
            cv2.aruco.drawDetectedMarkers(vis_image, corners, ids)
            if show:
                get_logger().debug("Displaying image...")
                cv2.imshow("Aruco Localization", vis_image)
                waitKey = cv2.waitKey(1)
                if waitKey & 0xFF == ord("q"):
                    get_logger().info("Quitting...")
                    self._is_okay = False
                    cv2.destroyAllWindows()
                elif waitKey & 0xFF == ord("s"):
                    get_logger().info("Saving image...")
                    filename = filename or "aruco_localization.png"
                    cv2.imwrite(filename, vis_image)
                if waitKey & 0xFF == ord(" "):
                    cv2.waitKey(0)
            if save:
                filename = filename or "aruco_localization.png"
                if Path(filename).suffix in [".png", ".jpg"]:
                    cv2.imwrite(filename, vis_image)
                else:
                    if not hasattr(self, "_writer"):
                        self._writer = VideoWriter(filename, 10, flush_interval=1)
                    self._writer.append(vis_image)

        # Estimate the pose of the markers
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self._marker_size,
            self._sensor.intrinsic_matrix,
            self._sensor.distortion_coefficients,
        )

        # Check that origin marker is detected
        ids_list = ids.flatten().tolist()
        if self._origin_id not in ids_list:
            get_logger().warning(f"Origin marker (ID {self._origin_id}) not detected.")
            return {}

        # Get origin pose
        origin_pose = self._get_pose(self._origin_id, ids_list, tvecs, rvecs)

        # Compute global poses for all specified markers
        results = {}
        for key, id in self._marker_ids.items():
            if id in ids_list:
                global_pose = self._get_global_pose(
                    origin_pose, id, ids_list, tvecs, rvecs
                )
                results[key] = global_pose

        results["image"] = image
        return results

    def _get_pose(self, id: int, ids: list, tvecs: np.ndarray, rvecs: np.ndarray):
        """Gets the pose of a marker with a specific ID.

        Args:
            id (int): The ID of the marker.
            ids (list): The list of detected marker IDs.
            tvecs (np.ndarray): The translation vectors of detected markers.
            rvecs (np.ndarray): The rotation vectors of detected markers.

        Returns:
            np.ndarray: The pose of the marker as [x, y, yaw].
        """
        idx = ids.index(id)
        tvec, rvec = tvecs[idx], rvecs[idx]
        rot = cv2.Rodrigues(rvec)[0]
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        return np.array([tvec[0, 0], tvec[0, 1], yaw])

    def _get_global_pose(
        self,
        origin_pose: np.ndarray,
        id: int,
        ids: list,
        tvecs: np.ndarray,
        rvecs: np.ndarray,
    ):
        """Computes the global pose of a marker relative to the origin.

        Args:
            origin_pose (np.ndarray): The pose of the origin marker.
            id (int): The ID of the target marker.
            ids (list): The list of detected marker IDs.
            tvecs (np.ndarray): The translation vectors of detected markers.
            rvecs (np.ndarray): The rotation vectors of detected markers.

        Returns:
            np.ndarray: The global pose of the marker as [x, y, yaw].
        """
        pose = origin_pose - self._get_pose(id, ids, tvecs, rvecs)
        pose[0] *= -1  # Flip x-axis
        return pose

    @property
    def is_okay(self) -> bool:
        """Checks if the algorithm and sensor are functioning properly.

        Returns:
            bool: True if both the algorithm and sensor are okay, False otherwise.
        """
        return self._is_okay and self._sensor.is_okay

    def close(self):
        """Closes resources associated with the algorithm.

        Closes any open writers and releases resources.
        """
        if hasattr(self, "_writer"):
            self._writer.close()
