from typing import Any
from dataclasses import field
from pathlib import Path

import pysurvive
import numpy as np
import pkg_resources

from cc_hardware.drivers.sensor import SensorData
from cc_hardware.drivers.mocap import MotionCaptureSensor, MotionCaptureSensorConfig
from cc_hardware.utils import config_wrapper, get_logger
from cc_hardware.utils.transformations import Frame, TransformationMatrix

# ===============


@config_wrapper
class ViveTrackerSensorConfig(MotionCaptureSensorConfig):
    """Config for the ViveTracker.

    Args:
        cfg (Path | str | None): Path to the config file. This should be a json file.

        additional_args (dict[str, Any]): Additional arguments to pass to the
            pysurvive.SimpleContext. The key should correspond to the argument passed
            to pysurvive but without the leading '--'. For example, to pass the argument
            '--poser MPFIT', the key should be 'poser' and the value should be 'MPFIT'.
    """

    cfg: Path | str | None = pkg_resources.resource_filename(
        "cc_hardware.drivers", str(Path("data") / "vive" / "config.json")
    )

    additional_args: dict[str, Any] = field(default_factory=dict)


# ===============


def SurvivePose_to_TransformationMatrix(
    pose: pysurvive.SurvivePose,
) -> TransformationMatrix:
    return Frame.create(
        pos=np.array(pose.Pos),
        quat=np.array([pose.Rot[1], pose.Rot[2], pose.Rot[3], pose.Rot[0]]),
    ).mat


class ViveTrackerPose(SensorData):
    def __init__(self):
        self.timestamp: float = 0
        self.mat: TransformationMatrix

    def process(self, data: pysurvive.SimpleObject):
        pose, timestamp = data.Pose()

        self.timestamp = timestamp
        self.mat = SurvivePose_to_TransformationMatrix(pose)

    def get_data(self) -> tuple[float, TransformationMatrix]:
        return self.timestamp, self.mat

    @staticmethod
    def read(data: pysurvive.SimpleObject) -> tuple[float, TransformationMatrix]:
        pose = ViveTrackerPose()
        pose.process(data)
        return pose.get_data()

# ===============


class ViveTrackerSensor(MotionCaptureSensor[ViveTrackerSensorConfig]):
    """"""

    def __init__(self, config: ViveTrackerSensorConfig):
        super().__init__(config)

        self._ctx = pysurvive.SimpleContext(self._get_argv())

    def _get_argv(self) -> list[str]:
        argv = []
        if self.config.cfg is not None:
            argv.extend(["-c", str(self.config.cfg)])
        for key, value in self.config.additional_args.items():
            argv.extend([f"--{key}", str(value)])
        return argv

    def accumulate(self, num_samples: int = 1) -> dict[str, tuple[float, TransformationMatrix]] | None:
        if not self.is_okay:
            get_logger().error("Vive tracker sensor is not okay")
            return

        data = {}
        for _ in range(num_samples):
            while (pose := self._ctx.NextUpdated()) is None:
                continue

            data[pose.Name().decode('utf-8')] = ViveTrackerPose.read(pose)

            for object in self._ctx.Objects():
                data[object.Name().decode('utf-8')] = ViveTrackerPose.read(object)

        return data

    @property
    def is_okay(self) -> bool:
        return self._ctx.Running()

    def close(self) -> None:
        pass
