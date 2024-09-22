import cv2
import numpy as np

from cc_hardware.algos.algo import Algorithm
from cc_hardware.drivers.camera import Camera
from cc_hardware.drivers.sensor import Sensor
from cc_hardware.utils.logger import get_logger


class ArucoLocalizationAlgorithm(Algorithm):
    def __init__(
        self,
        sensor: Sensor,
        *,
        aruco_dict: int,
        marker_size: float,
        origin_id: int,
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

    def run(self, *, visualize: bool = False, save: bool = False):
        """Processes a single image and returns the localization result."""
        results = []
        for _ in range(self._num_samples):
            results.append(self._process_image(visualize=visualize, save=save))

        # Average the results
        result = {}
        keys = set([key for r in results for key in r.keys()])
        for key in keys:
            if all(key in r for r in results):
                result[key] = np.median([r[key] for r in results], axis=0)

        return {key: result.get(key) for key in self._marker_ids}

    def _process_image(self, *, visualize: bool = False, save: bool = False) -> dict:
        """Process a single image to compute poses."""
        image = np.squeeze(self._sensor.accumulate(1))

        # Detect markers
        corners, ids, _ = self._detector.detectMarkers(image)
        if ids is None:
            raise RuntimeError("No markers detected.")

        # Visualize the results
        if visualize or save:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            if visualize:
                get_logger().debug("Displaying image...")
                cv2.imshow("Aruco Localization", image)
                waitKey = cv2.waitKey(1)
                if waitKey & 0xFF == ord("q"):
                    get_logger().info("Quitting...")
                    self._is_okay = False
                    cv2.destroyAllWindows()
                elif waitKey & 0xFF == ord("s"):
                    get_logger().info("Saving image...")
                    cv2.imwrite("aruco_localization.png", image)
            elif save:
                cv2.imwrite("aruco_localization.png", image)

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
        poses = {}
        for key, id in self._marker_ids.items():
            if id in ids_list:
                global_pose = self._get_global_pose(
                    origin_pose, id, ids_list, tvecs, rvecs
                )
                poses[key] = global_pose

        return poses

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
