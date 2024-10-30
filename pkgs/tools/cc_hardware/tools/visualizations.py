from functools import partial
from pathlib import Path

import cv2
import imageio
import numpy as np
import typer

try:
    import torch
except ImportError:
    torch = None

from cc_hardware.drivers.cameras.camera import Camera
from cc_hardware.drivers.spads.spad import SPADSensor
from cc_hardware.tools.app import APP
from cc_hardware.utils.constants import C
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.plotting import histogram_gui, plot_points, transient_gui

# ========================


def dashboard(spad: type[SPADSensor] | SPADSensor, **kwargs):
    from cc_hardware.utils.manager import Manager

    def setup(manager: Manager, spad: SPADSensor):
        histogram_gui(spad, **kwargs)

    with Manager(spad=spad) as manager:
        manager.run(setup=setup)


@APP.command()
def tmf8828_dashboard(
    port: str | None = None,
    num_frames: int = 100,
    show: bool = True,
    save: bool = False,
    filename: str | None = None,
    autoscale: bool = True,
    ylim: float | None = None,
    min_bin: int = 0,
    max_bin: int = 127,
    channel_mask: list[int] | None = None,
    spad_id: int = 6,  # 3x3
    short_range: bool = False,
    fullscreen: bool = False,
):
    from cc_hardware.drivers.spads.tmf8828 import RangeMode, TMF8828Sensor

    TMF8828Sensor.PORT = port or TMF8828Sensor.PORT

    assert spad_id in (
        6,
        7,
        15,
    ), f"Only 6 (3x3), 7 (4x4), and 15 (8x8) sensors are supported, got {spad_id}."
    range_mode = RangeMode.SHORT if short_range else RangeMode.LONG
    sensor = partial(TMF8828Sensor, spad_id=spad_id, range_mode=range_mode)

    dashboard(
        sensor,
        num_frames=num_frames,
        show=show,
        save=save,
        filename=filename,
        autoscale=autoscale,
        ylim=ylim,
        min_bin=min_bin,
        max_bin=max_bin,
        channel_mask=channel_mask,
        fullscreen=fullscreen,
    )


@APP.command()
def pkl_dashboard(
    pkl_path: Path,
    num_frames: int = 100,
    show: bool = True,
    save: bool = False,
    filename: str | None = None,
    autoscale: bool = True,
    ylim: float | None = None,
    min_bin: int = 0,
    max_bin: int = 127,
    channel_mask: list[int] | None = None,
    resolution: tuple[int, int] = (3, 3),
):
    from cc_hardware.drivers.spads.pkl import PklSPADSensor

    # TODO: Load directly from the pkl
    bin_width = 10 / 128 / C

    dashboard(
        PklSPADSensor(pkl_path, bin_width=bin_width, resolution=resolution),
        num_frames=num_frames,
        show=show,
        save=save,
        filename=filename,
        autoscale=autoscale,
        ylim=ylim,
        min_bin=min_bin,
        max_bin=max_bin,
        channel_mask=channel_mask,
    )


# ========================


def transient_viewer(
    sensor: type[SPADSensor] | SPADSensor,
    **kwargs,
):
    from cc_hardware.utils.manager import Manager

    def setup(manager: Manager, sensor: SPADSensor):
        transient_gui(sensor, **kwargs)

    with Manager(sensor=sensor) as manager:
        manager.run(setup=setup)


@APP.command()
def tmf8828_transient_viewer(
    port: str | None = None,
    show: bool = True,
    save: bool = False,
    filename: str | None = None,
    min_bin: int = 0,
    max_bin: int = 127,
    spad_id: int = 6,  # 3x3
    short_range: bool = False,
    fullscreen: bool = False,
    fps: int = 10,
    normalize_per_pixel: bool = True,
):
    from cc_hardware.drivers.spads.tmf8828 import RangeMode, TMF8828Sensor

    TMF8828Sensor.PORT = port or TMF8828Sensor.PORT

    assert spad_id in (
        6,
        7,
        15,
    ), f"Only 6 (3x3), 7 (4x4), and 15 (8x8) sensors are supported, got {spad_id}."
    range_mode = RangeMode.SHORT if short_range else RangeMode.LONG
    sensor = partial(TMF8828Sensor, spad_id=spad_id, range_mode=range_mode)

    transient_viewer(
        sensor,
        show=show,
        save=save,
        filename=filename,
        min_bin=min_bin,
        max_bin=max_bin,
        fullscreen=fullscreen,
        fps=fps,
        normalize_per_pixel=normalize_per_pixel,
    )


@APP.command()
def pkl_transient_viewer(
    pkl_path: Path,
    *,
    bin_width: float = typer.Option(..., "--bin-width", help="Bin width in meters"),
    res: tuple[int, int] = typer.Option(..., "--res", help="width, height"),
    min_bin: int = 0,
    max_bin: int = 127,
    fullscreen: bool = False,
    normalize_per_pixel: bool = True,
):
    from cc_hardware.drivers.spads.pkl import PklSPADSensor

    transient_viewer(
        PklSPADSensor(pkl_path, bin_width=bin_width, resolution=res),
        min_bin=min_bin,
        max_bin=max_bin,
        fullscreen=fullscreen,
        normalize_per_pixel=normalize_per_pixel,
    )


# ========================


def camera_viewer(
    camera: type[Camera] | Camera,
    num_frames: int,
    resolution: tuple[int, int] | None = None,
    **kwargs,
):
    from cc_hardware.utils.manager import Manager

    def setup(manager: Manager, camera: Camera):
        pass

    def loop(iter: int, manager: Manager, camera: Camera) -> bool:
        if num_frames != -1 and iter >= num_frames:
            get_logger().info(f"Finished capturing {num_frames} frames.")
            return False

        frame = camera.accumulate(num_samples=1)
        if frame is None:
            return False

        # Resize the frame
        if resolution is not None:
            frame = cv2.resize(frame, resolution)

        cv2.imshow("Camera Viewer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

        return True

    with Manager(camera=camera) as manager:
        manager.run(setup=setup, loop=loop)


@APP.command()
def flir_camera_viewer(num_frames: int = -1, resolution: tuple[int, int] | None = None):
    from cc_hardware.drivers.cameras.flir import GrasshopperFlirCamera

    camera_viewer(GrasshopperFlirCamera, num_frames, resolution)


@APP.command()
def pkl_camera_viewer(
    pkl_path: Path, num_frames: int = -1, resolution: tuple[int, int] | None = None
):
    from cc_hardware.drivers.cameras.pkl import PklCamera

    camera_viewer(PklCamera(pkl_path), num_frames, resolution)


@APP.command()
def realsense_camera_viewer(
    num_frames: int = -1,
    resolution: tuple[int, int] | None = None,
    rgb: bool | None = None,
    depth: bool | None = None,
    exposure: int | None = None,
):
    from cc_hardware.drivers.cameras.realsense import RealsenseCamera

    assert not (rgb and depth), "Cannot show both RGB and depth images."
    if depth:
        # Apply a colormap for visualization purposes
        class DepthRealsenseCamera(RealsenseCamera):
            def accumulate(self, num_samples: int):
                frame = super().accumulate(
                    num_samples=num_samples, return_rgb=False, return_depth=True
                )
                return cv2.applyColorMap(
                    cv2.convertScaleAbs(frame, alpha=0.03), cv2.COLORMAP_JET
                )

        RealsenseCamera = DepthRealsenseCamera

    if exposure is not None:
        RealsenseCamera = partial(RealsenseCamera, exposure=exposure)

    camera_viewer(RealsenseCamera, num_frames, resolution)


# ========================


def aruco_localization(
    camera: type[Camera] | Camera,
    num_frames: int,
    aruco_dict: str,
    marker_size: float,
    **kwargs,
):
    from cc_hardware.algos.aruco import ArucoLocalizationAlgorithm
    from cc_hardware.utils.manager import Manager

    assert hasattr(cv2.aruco, aruco_dict), f"Invalid aruco_dict: {aruco_dict}"
    aruco_dict = getattr(cv2.aruco, aruco_dict)

    def setup(manager: Manager, camera: Camera):
        algo = ArucoLocalizationAlgorithm(
            camera,
            aruco_dict=aruco_dict,
            marker_size=marker_size,
            origin_id=116,
        )
        manager.add(algo=algo)

    def loop(
        iter: int, manager: Manager, camera: Camera, algo: ArucoLocalizationAlgorithm
    ) -> bool:
        if num_frames != -1 and iter >= num_frames:
            get_logger().info(f"Finished capturing {num_frames} frames.")
            return False

        algo.run(**kwargs)

        return True

    with Manager(camera=camera) as manager:
        manager.run(setup=setup, loop=loop)


@APP.command()
def flir_aruco_localization(
    aruco_dict: str,
    marker_size: float = 8.25,
    show: bool = True,
    save: bool = False,
    filename: str | None = None,
    num_frames: int = -1,
):
    from cc_hardware.drivers.cameras.flir import GrasshopperFlirCamera

    aruco_localization(
        GrasshopperFlirCamera,
        aruco_dict=aruco_dict,
        marker_size=marker_size,
        show=show,
        save=save,
        filename=filename,
        num_frames=num_frames,
    )


@APP.command()
def pkl_aruco_localization(
    pkl_path: Path,
    aruco_dict: str,
    marker_size: float = 8.25,
    show: bool = True,
    save: bool = False,
    filename: str | None = None,
    num_frames: int = -1,
):
    from cc_hardware.drivers.cameras.pkl import PklCamera

    aruco_localization(
        PklCamera(pkl_path),
        aruco_dict=aruco_dict,
        marker_size=marker_size,
        show=show,
        save=save,
        filename=filename,
        num_frames=num_frames,
    )


# ========================


def estimated_position(
    sensor: type[SPADSensor] | SPADSensor,
    camera: type[Camera] | Camera,
    model: "torch.nn.Module",
    aruco_dict: str,
    marker_size: float = 8.25,
    num_samples: int = 1,
    show: bool = False,
    save: bool = False,
    filename: str | None = None,
):
    import matplotlib.pyplot as plt

    from cc_hardware.algos.aruco import ArucoLocalizationAlgorithm
    from cc_hardware.utils.manager import Manager

    def setup(manager: Manager, sensor: SPADSensor, camera: Camera):
        algo = ArucoLocalizationAlgorithm(
            camera,
            aruco_dict=aruco_dict,
            marker_size=marker_size,
            origin_id=116,
            camera_id=137,
            object_id=120,
        )
        manager.add(algo=algo)

    accumulated_points = {}
    fig = plt.figure(figsize=(5, 5))
    plt.xlim(56, 99)
    plt.ylim(75, 114)

    def loop(
        iter: int,
        manager: Manager,
        sensor: SPADSensor,
        camera: Camera,
        algo: ArucoLocalizationAlgorithm,
    ) -> bool:
        iter_filename = None
        if save and filename is not None:
            iter_filename = Path(filename).parent / "temp" / f"{iter:03d}.png"
            iter_filename.parent.mkdir(parents=True, exist_ok=True)

        histogram = sensor.accumulate(num_samples, average=True)
        if histogram is None:
            return False

        poses = algo.run(show=False, return_images=False)
        assert "camera_id" in poses, "No camera pose found."
        assert "object_id" in poses, "No object pose found."

        camera_pose = poses["camera_id"]
        object_pose = poses["object_id"]
        predicted_object_pose = model(histogram, camera_pose)

        get_logger().info(f"Camera pose: {camera_pose}")
        get_logger().info(f"Object pose: {object_pose}")
        get_logger().info(f"Predicted object pose: {predicted_object_pose}")

        accumulated_points.setdefault("camera", []).append(camera_pose[:2])
        accumulated_points.setdefault("object", []).append(object_pose[:2])
        accumulated_points.setdefault("predicted_object", []).append(
            predicted_object_pose[:2]
        )

        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.clf()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plot_points(
            *np.array(accumulated_points["object"]).T,
            *np.array(accumulated_points["predicted_object"]).T,
            fig=fig,
            filename=iter_filename,
        )

        if show:
            plt.pause(0.0001)

        return True

    def cleanup(*args, **kwargs):
        if save:
            assert filename is not None, "Filename must be provided for saving."
            save_filename = Path(filename)
            plt.savefig(save_filename.with_suffix(".png"))

            # Create animation
            images = []
            for image in sorted(save_filename.parent.glob("temp/*.png")):
                images.append(imageio.imread(image))
            imageio.mimwrite(save_filename, images, fps=10)

        if show:
            plt.show()
        plt.close()

    with Manager(sensor=sensor, camera=camera) as manager:
        manager.run(setup=setup, loop=loop, cleanup=cleanup)


@APP.command()
def pkl_estimated_position(
    pkl_path: Path,
    aruco_dict: str,
    marker_size: float = 8.25,
    num_samples: int = 1,
    show: bool = False,
    save: bool = False,
    filename: str | None = None,
):
    from cc_hardware.drivers.cameras.pkl import PklCamera
    from cc_hardware.drivers.spads.pkl import PklSPADSensor
    from cc_hardware.utils.writers import PklWriter

    assert hasattr(cv2.aruco, aruco_dict), f"Invalid aruco_dict: {aruco_dict}"
    aruco_dict = getattr(cv2.aruco, aruco_dict)

    assert torch is not None, "PyTorch is required for this command."

    class PklModel(torch.nn.Module):
        def __init__(self, pkl_path: Path):
            super().__init__()
            self._data = PklWriter.load_all(pkl_path)
            self._data_iterator = iter(self._data)

        def forward(self, histogram, camera_pose):
            if self._data_iterator is None:
                get_logger().error("No data available.")
                return None

            try:
                entry = next(self._data_iterator)
            except StopIteration:
                get_logger().error("No more data available.")
                self._data_iterator = None
                return None

            pose = entry["poses"]["object_id"]
            return pose + pose * np.random.randn(3) * 0.01

    model = PklModel(pkl_path)
    model.eval()

    estimated_position(
        PklSPADSensor(pkl_path, bin_width=10 / 128 / C, resolution=(3, 3)),
        PklCamera(pkl_path),
        model,
        aruco_dict=aruco_dict,
        marker_size=marker_size,
        num_samples=num_samples,
        show=show,
        save=save,
        filename=filename,
    )
