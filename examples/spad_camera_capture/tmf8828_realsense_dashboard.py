import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from cc_hardware.drivers.cameras.realsense import RealsenseConfig, RealsenseCamera
from cc_hardware.drivers.spads.tmf8828 import TMF8828Config, TMF8828Sensor, SPADID
from cc_hardware.drivers.spads.dashboards.matplotlib import (
    MatplotlibDashboard,
    MatplotlibDashboardConfig,
)
from cc_hardware.tools.cli import register_cli, run_cli
from cc_hardware.utils import get_logger
from cc_hardware.utils.manager import Manager, PrimitiveComponent
from cc_hardware.utils.file_handlers import PklHandler

warnings.filterwarnings("ignore")

now = datetime.now()

def process_depth(depth, max_depth_mm: int = 5000):
    return np.clip(depth.astype(np.float32) / max_depth_mm, 0, 1)

def process_rgb(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

def filter_detections(result, name, confidence_threshold):
    # Filter by name and confidence
    filtered_boxes = []
    for i, box in enumerate(result.boxes):
        if result.names[int(box.cls)] == name and box.conf >= confidence_threshold:
            filtered_boxes.append(box)

    # Return filtered boxes as a list
    return filtered_boxes

@register_cli
def camera_viewer(
    camera_config: RealsenseConfig = RealsenseConfig(start_pipeline_once=True),
    spad_config: TMF8828Config = TMF8828Config(spad_id=SPADID.ID6),
    dashboard_config: MatplotlibDashboardConfig = MatplotlibDashboardConfig(fullscreen=True),
):
    def setup(manager: Manager):
        camera = RealsenseCamera(camera_config)
        manager.add(camera=camera)

        spad = TMF8828Sensor(spad_config)
        manager.add(spad=spad)

        dashboard = MatplotlibDashboard(dashboard_config, spad)
        dashboard.setup()
        manager.add(dashboard=dashboard)

    def loop(
        iter: int,
        manager: Manager,
        camera: RealsenseCamera,
        spad: TMF8828Sensor,
        dashboard: MatplotlibDashboard,
    ) -> bool:
        get_logger().info(f"Capturing frame {iter}...")

        histogram = spad.accumulate()

        rgb, depth = camera.accumulate(return_rgb=True, return_depth=True)
        if rgb is None or depth is None:
            return False

        dashboard.update(iter, histograms=histogram)

        return True

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


if __name__ == "__main__":
    run_cli(camera_viewer)
