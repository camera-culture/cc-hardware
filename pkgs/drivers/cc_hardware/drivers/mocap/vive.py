from dataclasses import field
from pathlib import Path
from typing import Any
import threading
from datetime import datetime

import numpy as np
import pkg_resources
import pysurvive

from cc_hardware.drivers.mocap import MotionCaptureSensor, MotionCaptureSensorConfig
from cc_hardware.drivers.sensor import SensorData
from cc_hardware.utils import config_wrapper, get_logger
from cc_hardware.utils.transformations import Frame, TransformationMatrix
import pysurvive.pysurvive_generated

# ===============


@config_wrapper
class ViveTrackerSensorConfig(MotionCaptureSensorConfig):
    """Config for the ViveTracker.

    Args:
        cfg (Path | str | None): Path to the config file. This should be a json file.

        start_on_button (bool): If True, during initialization, the tracker will block
            until the button on the vive is pressed. Default is False.
        stop_on_button (bool): If True, during data acquisition, the button activation
            will be checked and the system will stop when the button is pressed.
        record_on_button (bool): If True, data is `only` recorded when a button is
            pressed. The accumulate method will essentially block until the button is
            pressed. If true, :attr:`start_on_button` continues to work as before,
            but :attr:`stop_on_button` has no effect. Default is False.

        additional_args (dict[str, Any]): Additional arguments to pass to the
            pysurvive.SimpleContext. The key should correspond to the argument passed
            to pysurvive but without the leading '--'. For example, to pass the argument
            '--poser MPFIT', the key should be 'poser' and the value should be 'MPFIT'.
    """

    cfg: Path | str | None = pkg_resources.resource_filename(
        "cc_hardware.drivers", str(Path("data") / "vive" / "config.json")
    )

    start_on_button: bool = False
    stop_on_button: bool = False
    record_on_button: bool = False

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
        # TODO: is there a better way, seems like i need this?
        # self.timestamp = float(datetime.now().timestamp())
        self.mat = SurvivePose_to_TransformationMatrix(pose)

    def get_data(self) -> tuple[float, TransformationMatrix]:
        return self.timestamp, self.mat

    @staticmethod
    def read(data: pysurvive.SimpleObject) -> tuple[float, TransformationMatrix]:
        pose = ViveTrackerPose()
        pose.process(data)
        return pose.get_data()


# ===============

_libs = pysurvive.pysurvive_generated._libs
POINTER = pysurvive.pysurvive_generated.POINTER
if _libs["survive"].has("survive_simple_get_ctx", "cdecl"):
    pysurvive.survive_simple_get_ctx = _libs["survive"].get(
        "survive_simple_get_ctx", "cdecl"
    )
    pysurvive.survive_simple_get_ctx.argtypes = [
        POINTER(pysurvive.SurviveSimpleContext)
    ]
    pysurvive.survive_simple_get_ctx.restype = POINTER(pysurvive.SurviveContext)

# ===============


class ViveTrackerSensor(MotionCaptureSensor[ViveTrackerSensorConfig]):
    """"""

    def __init__(self, config: ViveTrackerSensorConfig):
        super().__init__(config)

        self._ctx = pysurvive.SimpleContext(self._get_argv())
        self._full_ctx = pysurvive.survive_simple_get_ctx(self._ctx.ptr)

        def _button_callback(*args, **kwargs):
            get_logger().info("Received button press.")

            self._button_event.set()

        self._button_event = threading.Event()
        pysurvive.install_button_fn(self._full_ctx, _button_callback)

        if config.start_on_button:
            self._wait_for_button()
            get_logger().info("Starting Vive data capture.")

    def _wait_for_button(self):
        get_logger().info("Waiting for button to be pressed...")
        # Do two waits since down and up count as separate presses
        self._button_event.wait()
        self._button_event.clear()
        self._button_event.wait()
        self._button_event.clear()

    def _get_argv(self) -> list[str]:
        argv = []
        if self.config.cfg is not None:
            argv.extend(["-c", str(self.config.cfg)])
        for key, value in self.config.additional_args.items():
            argv.extend([f"--{key}", str(value)])
        return argv

    def accumulate(
        self, num_samples: int = 1
    ) -> dict[str, tuple[float, TransformationMatrix]] | None:
        if self.config.record_on_button:
            self._wait_for_button()
        elif self.config.stop_on_button and self._button_event.is_set():
            get_logger().info("Recording button press, stopping...")
            self.close()
            return

        data = {}
        good_samples = 0
        while good_samples < num_samples:
            if (pose := self._ctx.NextUpdated()) is None:
                continue

            try:
                data[pose.Name().decode("utf-8")] = ViveTrackerPose.read(pose)

                for obj in self._ctx.Objects():
                    data[obj.Name().decode("utf-8")] = ViveTrackerPose.read(obj)
            except ValueError as e:
                get_logger().warning(f"Got error while reading from vive tracker: {e}")
                continue

            good_samples += 1

        return data

    @property
    def is_okay(self) -> bool:
        return self._ctx.Running()

    def close(self) -> None:
        if hasattr(self, "_ctx") and self._ctx.Running():
            pysurvive.survive_simple_close(self._ctx.ptr)
        if hasattr(self, "_full_ctx"):
            pysurvive.survive_close(self._full_ctx)
