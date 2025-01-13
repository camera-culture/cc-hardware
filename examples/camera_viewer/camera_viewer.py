import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageGrab

from cc_hardware.drivers import Camera, CameraConfig
from cc_hardware.tools.cli import register_cli, run_cli
from cc_hardware.utils import get_logger
from cc_hardware.utils.manager import Manager, PrimitiveComponent

def get_display_size() -> tuple[int, int]:
    try:
        screen = ImageGrab.grab()  # Capture the virtual screen
        return screen.size  # Returns (width, height)
    except Exception as e:
        print(f"Error: {e}")
        return (0, 0)
WIDTH, HEIGHT = get_display_size()

@register_cli
def camera_viewer(
    camera: CameraConfig,
    num_frames: int = -1,
    resolution: tuple[int, int] | None = None,
    fullscreen: bool = False,
):

    def setup(manager: Manager):
        _camera = camera.create_instance()
        manager.add(camera=_camera)

        manager.add(start_time=PrimitiveComponent(time.time()))

    def loop(iter: int, manager: Manager, camera: Camera, start_time: PrimitiveComponent) -> bool:
        if num_frames != -1 and iter >= num_frames:
            get_logger().info(f"Finished capturing {num_frames} frames.")
            return False

        frame = camera.accumulate()
        if frame is None:
            return False

        # Resize the frame
        if resolution is not None:
            frame = cv2.resize(frame, resolution)

        # Calculate the FPS
        elapsed_time = time.time() - start_time.value
        fps = iter / elapsed_time
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if fullscreen:
            if iter == 0:
                cv2.namedWindow("Camera Viewer", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("Camera Viewer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        cv2.imshow("Camera Viewer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

        return True

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


if __name__ == "__main__":
    run_cli(camera_viewer)
