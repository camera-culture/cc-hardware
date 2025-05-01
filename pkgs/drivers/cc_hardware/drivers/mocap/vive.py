# from dataclasses import field
# from pathlib import Path
# from typing import Any
# import threading
# from datetime import datetime

# import numpy as np
# import pkg_resources
# import pysurvive

# from cc_hardware.drivers.mocap import MotionCaptureSensor, MotionCaptureSensorConfig
# from cc_hardware.drivers.sensor import SensorData
# from cc_hardware.utils import config_wrapper, get_logger
# from cc_hardware.utils.transformations import Frame, TransformationMatrix
# import pysurvive.pysurvive_generated

# # ===============


# @config_wrapper
# class ViveTrackerSensorConfig(MotionCaptureSensorConfig):
#     """Config for the ViveTracker.

#     Args:
#         cfg (Path | str | None): Path to the config file. This should be a json file.

#         start_on_button (bool): If True, during initialization, the tracker will block
#             until the button on the vive is pressed. Default is False.
#         stop_on_button (bool): If True, during data acquisition, the button activation
#             will be checked and the system will stop when the button is pressed.
#         record_on_button (bool): If True, data is `only` recorded when a button is
#             pressed. The accumulate method will essentially block until the button is
#             pressed. If true, :attr:`start_on_button` continues to work as before,
#             but :attr:`stop_on_button` has no effect. Default is False.

#         objects (dict[str, str] | None): A dictionary of vive object ids to names that
#             will be returned in the pose data.

#         additional_args (dict[str, Any]): Additional arguments to pass to the
#             pysurvive.SimpleContext. The key should correspond to the argument passed
#             to pysurvive but without the leading '--'. For example, to pass the argument
#             '--poser MPFIT', the key should be 'poser' and the value should be 'MPFIT'.
#     """

#     cfg: Path | str | None = pkg_resources.resource_filename(
#         "cc_hardware.drivers", str(Path("data") / "vive" / "config.json")
#     )

#     start_on_button: bool = False
#     stop_on_button: bool = False
#     record_on_button: bool = False

#     objects: dict[str, str] | None = None

#     additional_args: dict[str, Any] = field(default_factory=dict)


# # ===============


# def SurvivePose_to_TransformationMatrix(
#     pose: pysurvive.SurvivePose,
# ) -> TransformationMatrix:
#     return Frame.create(
#         pos=np.array(pose.Pos),
#         quat=np.array([pose.Rot[1], pose.Rot[2], pose.Rot[3], pose.Rot[0]]),
#     ).mat


# class ViveTrackerPose(SensorData):
#     def __init__(self):
#         self.timestamp: float = 0
#         self.mat: TransformationMatrix

#     def process(self, data: pysurvive.SimpleObject):
#         pose, timestamp = data.Pose()

#         self.timestamp = timestamp
#         # TODO: is there a better way, seems like i need this?
#         # self.timestamp = float(datetime.now().timestamp())
#         self.mat = SurvivePose_to_TransformationMatrix(pose)

#     def get_data(self) -> tuple[float, TransformationMatrix]:
#         return self.timestamp, self.mat

#     @staticmethod
#     def read(data: pysurvive.SimpleObject) -> tuple[float, TransformationMatrix]:
#         pose = ViveTrackerPose()
#         pose.process(data)
#         return pose.get_data()


# # ===============

# _libs = pysurvive.pysurvive_generated._libs
# POINTER = pysurvive.pysurvive_generated.POINTER
# if _libs["survive"].has("survive_simple_get_ctx", "cdecl"):
#     pysurvive.survive_simple_get_ctx = _libs["survive"].get(
#         "survive_simple_get_ctx", "cdecl"
#     )
#     pysurvive.survive_simple_get_ctx.argtypes = [
#         POINTER(pysurvive.SurviveSimpleContext)
#     ]
#     pysurvive.survive_simple_get_ctx.restype = POINTER(pysurvive.SurviveContext)

# # ===============


# class ViveTrackerSensor(MotionCaptureSensor[ViveTrackerSensorConfig]):
#     """"""

#     def __init__(self, config: ViveTrackerSensorConfig):
#         super().__init__(config)

#         self._ctx = pysurvive.SimpleContext(self._get_argv())
#         self._full_ctx = pysurvive.survive_simple_get_ctx(self._ctx.ptr)

#         def _button_callback(*args, **kwargs):
#             get_logger().info("Received button press.")

#             self._button_event.set()

#         self._button_event = threading.Event()
#         pysurvive.install_button_fn(self._full_ctx, _button_callback)

#         if config.start_on_button:
#             self._wait_for_button()
#             get_logger().info("Starting Vive data capture.")

#     def _wait_for_button(self):
#         get_logger().info("Waiting for button to be pressed...")
#         # Do two waits since down and up count as separate presses
#         self._button_event.wait()
#         self._button_event.clear()
#         self._button_event.wait()
#         self._button_event.clear()

#     def _get_argv(self) -> list[str]:
#         argv = []
#         if self.config.cfg is not None:
#             argv.extend(["-c", str(self.config.cfg)])
#         for key, value in self.config.additional_args.items():
#             argv.extend([f"--{key}", str(value)])
#         return argv

#     def accumulate(
#         self, num_samples: int = 1
#     ) -> dict[str, tuple[float, TransformationMatrix]] | None:
#         if self.config.record_on_button:
#             self._wait_for_button()
#         elif self.config.stop_on_button and self._button_event.is_set():
#             get_logger().info("Recording button press, stopping...")
#             self.close()
#             return

#         data = {}
#         good_samples = 0
#         while good_samples < num_samples:
#             if (pose := self._ctx.NextUpdated()) is None:
#                 continue

#             try:
#                 data[pose.Name().decode("utf-8")] = ViveTrackerPose.read(pose)

#                 for obj in self._ctx.Objects():
#                     data[obj.Name().decode("utf-8")] = ViveTrackerPose.read(obj)
#             except ValueError as e:
#                 get_logger().warning(f"Got error while reading from vive tracker: {e}")
#                 for obj in self._ctx.Objects():
#                     obj.Pose()
#                 continue

#             good_samples += 1

#         if self.config.objects:
#             data = {
#                 self.config.objects[v]: data[k]
#                 for k, v in data.items()
#                 if k in self.config.objects
#             }

#         return data

#     @property
#     def is_okay(self) -> bool:
#         return self._ctx.Running()

#     def close(self):
#         pass


from dataclasses import field
from pathlib import Path
from typing import Any, Optional
import multiprocessing
from importlib.resources import files

import numpy as np
import pysurvive
import yaml

from cc_hardware.drivers.mocap import MotionCaptureSensor, MotionCaptureSensorConfig
from cc_hardware.drivers.sensor import SensorData
from cc_hardware.utils import config_wrapper, get_logger
from cc_hardware.utils.transformations import Frame, FrameConfig, TransformationMatrix
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

        objects (dict[str, str | None]): A dictionary of serial numbers to object names.
            If non-empty, only the objects with the serial numbers in this dictionary
            are recorded. An optional object name can be provided which maps replaces
            the serial number in the output dictionary.

        additional_args (dict[str, Any]): Additional arguments to pass to the
            pysurvive.SimpleContext. The key should correspond to the argument passed
            to pysurvive but without the leading '--'. For example, to pass the argument
            '--poser MPFIT', the key should be 'poser' and the value should be 'MPFIT'.
    """

    cfg: Path | str | None = files("cc_hardware.drivers").joinpath(
        "data", "vive", "config.json"
    )

    start_on_button: bool = False
    stop_on_button: bool = False
    record_on_button: bool = False

    objects: dict[str, str | None] = field(default_factory=dict)

    origin: FrameConfig = field(default_factory=FrameConfig)

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
    def read(
        data: pysurvive.SimpleObject, L: TransformationMatrix, R: TransformationMatrix
    ) -> tuple[float, TransformationMatrix]:
        pose = ViveTrackerPose()
        pose.process(data)
        timestamp, mat = pose.get_data()
        mat = L @ mat @ R
        return timestamp, mat


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

        self._objects: dict[str, str] = {
            k: v for k, v in config.objects.items() if v is not None
        }

        self._initialize_calibration()
        self._initialize_context()
        self._initialize_button()

    def _initialize_calibration(self):
        try:
            filename = files("cc_hardware.drivers").joinpath(
                "data", "vive", "origin.yaml"
            )
            assert filename.exists(), f"Calibration file {filename} does not exist."
            origin_calibration = Frame.from_yaml(filename)
        except Exception as e:
            get_logger().warning(
                f"Could not load origin calibration! Setting to identity."
            )
            origin_calibration = Frame.create()

        self._calibration = origin_calibration

    def _initialize_context(self):
        self._ctx = pysurvive.SimpleContext(self._get_argv())
        self._full_ctx = pysurvive.survive_simple_get_ctx(self._ctx.ptr)

    def _initialize_button(self):
        if not (
            self.config.start_on_button
            or self.config.record_on_button
            or self.config.stop_on_button
        ):
            return

        self._button_event = multiprocessing.Event()
        pysurvive.install_button_fn(self._full_ctx, self._button_callback)

        if self.config.start_on_button:
            get_logger().info("Waiting for button press to start.")
            self._wait_for_button()
            get_logger().info("Starting Vive data capture.")

    def _button_callback(self, *args, **kwargs):
        get_logger().info("Received button press.")
        self._button_event.set()

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

        # Wait for the next update
        pysurvive.survive_simple_wait_for_update(self._ctx.ptr)

        data = {}
        good_samples = 0
        while good_samples < num_samples:
            try:
                obj = pysurvive.survive_simple_get_first_object(self._ctx.ptr)
                if not obj:
                    get_logger().warning("No objects found.")
                    continue

                while obj:
                    serial = pysurvive.survive_simple_serial_number(obj).decode("utf-8")
                    if self._objects and serial not in self._objects:
                        obj = pysurvive.survive_simple_get_next_object(
                            self._ctx.ptr, obj
                        )
                        continue

                    name = self._objects.get(serial, serial)

                    # Create the transformation matrix for the object
                    L = self._calibration.inverse().mat
                    R = Frame.create().mat

                    data[name] = ViveTrackerPose.read(pysurvive.SimpleObject(obj), L, R)
                    obj = pysurvive.survive_simple_get_next_object(self._ctx.ptr, obj)
            except ValueError as e:
                get_logger().debug(f"Got error while reading from vive tracker: {e}")
                continue

            good_samples += 1

        return data

    def calibrate(self) -> bool:
        import time, tqdm

        # First close the tracker
        get_logger().info("Closing the tracker.")
        self.close()

        # Remove ~/.config/libsurvive/config.json and calibration
        get_logger().info("Removing ~/.config/libsurvive/config.json")
        Path.home().joinpath(".config", "libsurvive", "config.json").unlink(
            missing_ok=True
        )
        self._calibration = Frame.create()

        # Restart the tracker
        get_logger().info("Restarting the tracker.")
        self._ctx = pysurvive.SimpleContext(self._get_argv())
        self._full_ctx = pysurvive.survive_simple_get_ctx(self._ctx.ptr)

        # Get the first pose
        get_logger().info("Getting the first pose.")
        data = self.accumulate(20)
        frames = {n: Frame.create(mat=mat).to_dict() for n, (_, mat) in data.items()}

        # Parse frames to only use the names prefixed with LHR (trackers)
        # TODO: any more generic way to do this?
        frames = {k: v for k, v in frames.items() if k.startswith("LHR")}
        assert len(frames) == 1, f"Only one tracker should be visible for calibration, found {len(frames)}."
        frame = list(frames.values())[0]

        # Save the initial pose to a file in the data dir
        get_logger().info("Saving the initial pose to a file.")
        filename = files("cc_hardware.drivers").joinpath("data", "vive", "origin.yaml")
        get_logger().info(f"Saving to {filename}.")
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as f:
            yaml.dump(frame, f)

        get_logger().info("Calibration complete.")

        return True

    @property
    def is_okay(self) -> bool:
        return self._ctx.Running()

    def close(self) -> None:
        if hasattr(self, "_ctx") and self._ctx.Running():
            get_logger().info("Closing Vive tracker...")
            pysurvive.survive_simple_close(self._ctx.ptr)
            del self._ctx
            get_logger().info("Vive tracker closed.")
