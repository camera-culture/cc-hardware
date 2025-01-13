import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings
from datetime import datetime
from pathlib import Path
import time

# from ultralytics import YOLO

from cc_hardware.drivers.cameras.realsense import RealsenseConfig, RealsenseCamera
from cc_hardware.drivers.spads.spad import SPADSensorConfig, SPADSensor
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
    pkl_name: str,
    spad_config: SPADSensorConfig,
    camera_config: RealsenseConfig = RealsenseConfig(start_pipeline_once=True),
    dashboard_config: MatplotlibDashboardConfig = MatplotlibDashboardConfig(fullscreen=True),
    logdir: Path = Path("logs") / now.strftime("%Y-%m-%d"),
    overwrite: bool = False,
):
    cv2.namedWindow("rgb", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("rgb", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def setup(manager: Manager):
        logdir.mkdir(parents=True, exist_ok=True)

        if (logdir / pkl_name).exists() and not overwrite:
            raise FileExistsError(f"File {logdir / pkl_name} already exists.")
        manager.add(writer=PklHandler(logdir / pkl_name))

        # Plot image and histogram next to each other
        fig, ax = plt.subplots(2, 1, figsize=(1, 1))

        # Create blank image to start out
        rgb_image = ax[0].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        ax[0].grid(False)
        ax[0].axis("off")
        depth_image = ax[1].imshow(np.zeros((480, 640)), cmap="jet_r", vmin=0, vmax=1)
        ax[1].grid(False)
        ax[1].axis("off")

        manager.add(rgb_image_component=PrimitiveComponent(rgb_image))
        manager.add(depth_image_component=PrimitiveComponent(depth_image))

        # model = YOLO("yolo11n-seg_ncnn_model")
        # manager.add(model_component=PrimitiveComponent(model))
        manager.add(model_component=None)

        camera = RealsenseCamera(camera_config)
        manager.add(camera=camera)

        spad = spad_config.create_instance()
        manager.add(spad=spad)

        # dashboard = MatplotlibDashboard(dashboard_config, spad)
        # dashboard.setup(fig)
        # manager.add(dashboard=dashboard)
        manager.add(dashboard=None)

        input("Ready to record ambient...")
        time.sleep(5)
        histograms = spad.accumulate(10, average=False)
        manager.components["writer"].append({"ambients": histograms, "resolution": spad.resolution, "num_bins": spad.num_bins})
        input("Ready to start capture...")
        time.sleep(2)

    def loop(
        iter: int,
        manager: Manager,
        writer: PklHandler,
        camera: RealsenseCamera,
        spad: SPADSensor,
        dashboard: MatplotlibDashboard,
        model_component: PrimitiveComponent,
        rgb_image_component: PrimitiveComponent,
        depth_image_component: PrimitiveComponent
    ) -> bool:
        get_logger().info(f"Capturing frame {iter}...")

        rgb_image = rgb_image_component.value
        depth_image = depth_image_component.value
        # model: YOLO = model_component.value

        t0 = time.time()
        rgb, depth = camera.accumulate(return_rgb=True, return_depth=True)
        if rgb is None or depth is None:
            return False
        t1 = time.time()

        rgb_image.set_data(process_rgb(rgb))
        t2 = time.time()
        depth_image.set_data(process_depth(depth))
        t3 = time.time()

        histogram = spad.accumulate()
        t3 = time.time()
        # dashboard.update(iter, histograms=histogram)
        cv2.imshow("rgb", rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        t4 = time.time()

        # result = model.predict(rgb, classes=[0], conf=0.4)[0]

        # rgb_plot = result.plot()
        # rgb_image.set_data(process_rgb(rgb_plot))
        # depth_image.set_data(process_depth(depth))

        # masks = result.masks
        # if masks is None:
        #     get_logger().warning("No masks found.")
        #     return True

        # pixel_coords = masks.xy
        # if len(pixel_coords) > 1:
        #     get_logger().warning(f"Expected 1 mask, got {len(pixel_coords)}")
        #     return True

        # pixel_coords = np.clip(pixel_coords[0], [0, 0], [639, 479]).astype(int)
        # person_depth = np.median(depth[pixel_coords[:, 1], pixel_coords[:, 0]])
        # get_logger().info(f"Person depth: {person_depth}")

        if iter > 5:
            writer.append(
                {
                    "iter": iter,
                    # "person_depth": person_depth,
                    "histogram": histogram,
                    "rgb": rgb,
                    "depth": depth,
                    # "masks": masks,
                }
            )
        t5 = time.time()

        get_logger().info(f"{iter}: \n\tCapture time: {t1 - t0:.2f}s \n\tProcessing time: {t3 - t2:.2f}s \n\tDashboard time: {t4 - t3:.2f}s \n\tWrite time: {t5 - t4:.2f}s \n\tTotal time: {t5 - t0:.2f}s")

        return True

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


if __name__ == "__main__":
    run_cli(camera_viewer)
