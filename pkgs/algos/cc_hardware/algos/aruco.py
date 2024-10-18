from pathlib import Path

import cv2
import numpy as np

from cc_hardware.algos.algo import Algorithm
from cc_hardware.drivers.camera import Camera
from cc_hardware.drivers.sensor import Sensor
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.writers import VideoWriter


class ArucoLocalizationAlgorithm(Algorithm):
    def __init__(
        self,
        sensor: Sensor,
        *,
        aruco_dict: int,
        marker_size: float,
        origin_id: int = -1,
        num_samples: int = 1,
        **marker_ids,
    ):
        super().__init__(sensor)

        assert isinstance(sensor, Camera), "Aruco algo requires a camera sensor."
        self._sensor: Camera

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
        """Processes a single image and returns the localization result."""
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
        """Process a single image to compute poses."""
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
        pose = origin_pose - self._get_pose(id, ids, tvecs, rvecs)
        pose[0] *= -1  # Flip x-axis
        return pose

    @property
    def is_okay(self) -> bool:
        return self._is_okay and self._sensor.is_okay

    def close(self):
        if hasattr(self, "_writer"):
            self._writer.close()
