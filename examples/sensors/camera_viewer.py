import cv2

from cc_hardware.drivers import Camera, CameraConfig
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli


@register_cli
def camera_viewer(
    camera: CameraConfig,
    num_frames: int = -1,
    resolution: tuple[int, int] | None = None,
):
    def setup(manager: Manager):
        _camera = Camera.create_from_config(camera)
        manager.add(camera=_camera)

    def loop(iter: int, manager: Manager, camera: Camera) -> bool:
        if num_frames != -1 and iter >= num_frames:
            get_logger().info(f"Finished capturing {num_frames} frames.")
            return False

        frame = camera.accumulate()
        if frame is None:
            return False

        # Resize the frame
        if resolution is not None:
            frame = cv2.resize(frame, resolution)

        cv2.imshow("Camera Viewer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

        return True

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


if __name__ == "__main__":
    run_cli(camera_viewer)
