import pysurvive
import numpy as np

from cc_hardware.drivers.sensor import SensorData
from cc_hardware.drivers.mocap import MotionCaptureSensor, MotionCaptureSensorConfig
from cc_hardware.utils import config_wrapper, get_logger

# ===============

@config_wrapper
class ViveTrackerSensorConfig(MotionCaptureSensorConfig):
    pass

# ===============


class ViveTrackerPose(SensorData):
    def __init__(self, config: ViveTrackerSensorConfig):
        self._config = config
        self.pos: np.ndarray = np.zeros(3)
        self.rot: np.ndarray = np.zeros(4)
        self.timestamp: float = 0

    def process(self, data: pysurvive.SimpleObject):
        pose, timestamp = data.Pose()
        self.pos[:] = pose.Pos
        self.rot[:] = pose.Rot
        self.timestamp = timestamp

    def get_data(self) -> tuple[float, np.ndarray, np.ndarray]:
        return self.timestamp, self.pos.copy(), self.rot.copy()

# ===============

class ViveTrackerSensor(MotionCaptureSensor[ViveTrackerSensorConfig]):
    def __init__(self, config: ViveTrackerSensorConfig):
        super().__init__(config)

        self._ctx = pysurvive.SimpleContext([])
        self._pose = ViveTrackerPose(config)

    def accumulate(self, num_samples: int = 1) -> np.ndarray | None:
        if not self.is_okay:
            get_logger().error("Vive tracker sensor is not okay")
            return

        poses = []
        for _ in range(num_samples):
            while (pose := self._ctx.NextUpdated()) is None:
                continue

            self._pose.process(pose)
            _, *pose = self._pose.get_data()
            poses.append(pose)

        return poses[0] if num_samples == 1 else poses

    @property
    def is_okay(self) -> bool:
        return self._ctx.Running()

    def close(self) -> None:
        pass