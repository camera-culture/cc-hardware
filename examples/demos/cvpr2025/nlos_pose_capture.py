import time
from datetime import datetime
from functools import partial
from pathlib import Path

import tqdm
import cv2
import torch
from ultralytics import YOLO

from cc_hardware.drivers.spads import SPADDataType, SPADSensor, SPADSensorConfig
from cc_hardware.drivers.spads.spad_wrappers import SPADMovingAverageWrapperConfig
from cc_hardware.drivers.spads.vl53l8ch import RangingMode, VL53L8CHConfig8x8
from cc_hardware.drivers.cameras import Camera
from cc_hardware.drivers.cameras.realsense import RealsenseConfig
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli, threaded_component, AtomicVariable
from cc_hardware.utils.file_handlers import PklHandler

# ==========

NOW = datetime.now()
LOGDIR: Path = Path("logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S")
OUTPUT_PKL: Path = LOGDIR / "data.pkl"

WRAPPED_SENSOR = VL53L8CHConfig8x8.create(
    num_bins=18,
    subsample=2,
    start_bin=30,
    ranging_mode=RangingMode.CONTINUOUS,
    ranging_frequency_hz=1,
    data_type=SPADDataType.HISTOGRAM | SPADDataType.POINT_CLOUD | SPADDataType.DISTANCE,
)
# WRAPPED_SENSOR = SPADBackgroundRemovalWrapperConfig.create(
#     pkl_spad=PklSPADSensorConfig.create(
#         pkl_path=Path("logs") / "2025-06-05/11-26-31/data.pkl",
#         index=1,
#     ),
#     wrapped=WRAPPED_SENSOR,
# )
WRAPPED_SENSOR = SPADMovingAverageWrapperConfig.create(
    wrapped=WRAPPED_SENSOR,
    window_size=1,
)
SENSOR = WRAPPED_SENSOR

CAMERA = RealsenseConfig.create(align=False)

NUM_SAMPLES = 100

# ==========

HISTOGRAM = AtomicVariable(None)

def sensor_callback(
    future,
    *,
    manager: Manager,
    sensor: SPADSensor,
):
    if not manager.is_looping:
        get_logger().info("Manager is not looping, stopping stepper callback.")
        return

    data = future.result()
    assert SPADDataType.HISTOGRAM in data, "Sensor must support histogram data type."
    HISTOGRAM.set(data[SPADDataType.HISTOGRAM])

    sensor.accumulate().add_done_callback(
        partial(
            sensor_callback,
            manager=manager,
            sensor=sensor,
        )
    )

# ==========


def setup(
    manager: Manager,
    sensor: SPADSensorConfig,
    camera: RealsenseConfig,
    record: bool = False,
    sensor_port: str | None = None,
    background: bool = True,
):
    """Configures the manager with sensor and dashboard instances.

    Args:
        manager (Manager): Manager to add sensor and dashboard to.
    """
    if record:
        LOGDIR.mkdir(exist_ok=True, parents=True)

        OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
        assert not OUTPUT_PKL.exists(), f"Output file {OUTPUT_PKL} already exists"
        writer = PklHandler(OUTPUT_PKL)
        manager.add(writer=writer)

        writer.append(dict(config=sensor.to_dict()))

    _sensor = threaded_component(SPADSensor.create_from_config(sensor, port=sensor_port))
    manager.add(sensor=_sensor)

    _sensor.accumulate().add_done_callback(
        partial(
            sensor_callback,
            manager=manager,
            sensor=_sensor,
        )
    )

    _camera: Camera = camera.create_from_registry(config=camera)
    manager.add(camera=_camera)

    algorithm = YOLO("yolo11s-pose.pt")
    algorithm.model.to("cuda" if torch.cuda.is_available() else "cpu")
    manager.add(algorithm=algorithm)

    pbar = tqdm.tqdm(
        total=NUM_SAMPLES,
        desc="Frames...",
        leave=False,
    )
    manager.add(pbar=pbar)

    # If background, capture N frames with frame = -1
    if background:
        input("Press Enter to start background capture...")
        for _ in range(10):
            data = _sensor.accumulate()
            if writer is not None:
                writer.append({"iter": -1, **data})
        input("Background capture complete. Press Enter to continue...")

    LOGDIR.mkdir(exist_ok=True, parents=True)
    PklHandler(LOGDIR / "config.pkl").write(
        dict(
            sensor=sensor,
            camera=camera,
            pkl_path=OUTPUT_PKL,
        )
    )


def loop(
    iter: int,
    manager: Manager,
    sensor: SPADSensor,
    camera: Camera,
    pbar: tqdm.tqdm,
    algorithm: YOLO,
    writer: PklHandler | None = None,
):
    if iter >= NUM_SAMPLES:
        return False

    histogram = HISTOGRAM.get()
    if histogram is None:
        manager.set_iter(iter - 1)
        return

    image = camera.accumulate()

    results = algorithm.predict(source=image, verbose=False)
    result = results[0].cpu()
    if result is None or result.keypoints is None:
        get_logger().warning("No keypoints detected in the image.")
        return
    frame = result.plot()

    cv2.imshow("Camera Image", frame)
    cv2.waitKey(1)

    if writer is not None:
        writer.append(
            {
                "iter": iter,
                "histogram": histogram,
                "keypoints": result.keypoints,
            }
        )

    pbar.update(1)

@register_cli
def nlos_pose_capture(
    sensor_port: str | None = None,
    record: bool = False,
    background: bool = True,
):
    """Sets up and runs the SPAD dashboard.

    Args:
        sensor (SPADSensorConfig): Configuration object for the sensor.
        dashboard (SPADDashboardConfig): Configuration object for the dashboard.
    """

    global t0
    t0 = time.time()

    with Manager() as manager:
        manager.run(
            setup=partial(
                setup,
                sensor_port=sensor_port,
                sensor=SENSOR,
                camera=CAMERA,
                record=record,
                background=background,
            ),
            loop=loop,
        )


if __name__ == "__main__":
    run_cli(nlos_pose_capture)
