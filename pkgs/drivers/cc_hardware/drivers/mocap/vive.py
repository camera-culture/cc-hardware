from typing import Any
from dataclasses import field, dataclass
from pathlib import Path
import ctypes

import pysurvive
import numpy as np
import pkg_resources

from cc_hardware.drivers.sensor import SensorData
from cc_hardware.drivers.mocap import MotionCaptureSensor, MotionCaptureSensorConfig
from cc_hardware.utils import config_wrapper, get_logger

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

dataclass(slots=True, kw_only=True)


class ViveTrackerPose:
    id: int
    timestamp: int
    pos: np.ndarray
    rot: np.ndarray


class ViveTrackerData(SensorData):
    def __init__(self, config: ViveTrackerSensorConfig):
        self._config = config
        self._trackers: dict[str, ViveTrackerPose] = {}

        self._pose = pysurvive.SurvivePose()
        self._pos = np.empty(3, dtype=np.float32)
        self._rot = np.empty(4, dtype=np.float32)

    def process(self, event: pysurvive.struct_SurviveSimpleEvent):
        if event.event_type == pysurvive.SurviveSimpleEventType_PoseUpdateEvent:
            self._process_pose_update_event(event)
        elif event.event_type == pysurvive.SurviveSimpleEventType_DeviceAdded:
            self._process_device_added(event)
        else:
            get_logger().warning(f"Got unrecognized event type: {event.event_type}")
            return

    def _process_pose_update_event(
        self, event: pysurvive.struct_SurviveSimpleObjectEvent
    ):
        pose_event: pysurvive.struct_SurviveSimpleObjectEvent = (
            pysurvive.survive_simple_get_pose_updated_event(event)
        )

        if (
            pysurvive.survive_simple_object_get_type(pose_event.object)
            == pysurvive.SurviveSimpleObject_LIGHTHOUSE
        ):
            get_logger().warning(
                "Got pose update event for lighthouse object. Ignoring..."
            )

        tracker_id = pysurvive.survive_simple_serial_number(pose_event.object)
        if tracker_id not in self._trackers:
            get_logger().info(f"Got new tracker: {tracker_id}.")

        timestamp = pysurvive.survive_simple_object_get_latest_pose(
            pose_event.object, self._pose
        )
        self._pos[:] = self._pose.Pos
        self._rot[:] = self._pose.Rot
        self._trackers[tracker_id] = ViveTrackerPose(
            tracker_id, timestamp, self._pos, self._rot
        )

    def _process_device_added(self, event: pysurvive.SurviveSimpleObjectEvent):
        #   const struct SurviveSimpleObjectEvent * object_event = survive_simple_get_object_event(
            # &event);
        pysurvive.survive_simple_get_object_event(event)

    def get_data(
        self, *, tracker_id: int | None = None, assume_one_tracker: bool = True
    ) -> dict[str, ViveTrackerPose] | ViveTrackerPose | None:
        if not self._has_data:
            return None

        if tracker_id is not None:
            assert tracker_id in self._trackers
            return self._trackers[tracker_id]
        elif assume_one_tracker:
            assert len(self._trackers) == 1
            return list(self._trackers.values())[0]
        else:
            return self._trackers

    def reset(self):
        super().reset()
        self._trackers.clear()

    @property
    def has_data(self) -> bool:
        return len(self._trackers) > 0

# ===============


class ViveTrackerSensor(MotionCaptureSensor[ViveTrackerSensorConfig]):
    def __init__(self, config: ViveTrackerSensorConfig):
        super().__init__(config)

        self._event = pysurvive.SurviveSimpleEvent()
        self._data = ViveTrackerData(config)

        self._ctx = pysurvive.simple_init(*self._get_argv())
        pysurvive.survive_simple_start_thread(self._ctx)

    def _get_argv(self) -> tuple[list[str], list[str]]:
        sargv = []
        if self.config.cfg is not None:
            sargv.extend(["-c", str(self.config.cfg)])
        for key, value in self.config.additional_args.items():
            sargv.extend([f"--{key}", str(value)])

        argc = len(sargv)
        argv = (pysurvive.LP_c_char * (argc + 1))()
        for i, arg in enumerate(sargv):
            enc_arg = arg.encode('utf-8')
            argv[i] = ctypes.create_string_buffer(enc_arg)
        return argc, argv

    def accumulate(self, num_samples: int = 1) -> np.ndarray | None:
        if not self.is_okay:
            get_logger().error("Vive tracker sensor is not okay")
            return

        poses = []
        for _ in range(num_samples):
            if (
                pysurvive.survive_simple_wait_for_event(self._ctx, self._event)
                == pysurvive.SurviveSimpleEventType_Shutdown
            ):
                get_logger().warning("Vive got shutdown event.")
                return

            while not self._data.has_data:
                self._data.process(self._event)
            poses.append(self._data.get_data())
            self._data.reset()

        return poses[0] if num_samples == 1 else poses

    @property
    def is_okay(self) -> bool:
        return pysurvive.survive_simple_is_running(self._ctx)

    def close(self) -> None:
        if hasattr(self, "_ctx"):
            pysurvive.survive_simple_close(self._ctx)
