from pathlib import Path

import numpy as np
import cv2

from ultralytics import YOLO
from ultralytics.utils import ops
import matplotlib.pyplot as plt

from cc_hardware.drivers.spads.pkl import PklSPADSensorConfig, PklSPADSensor
from cc_hardware.drivers.spads import SPADDashboardConfig, SPADDashboard
from cc_hardware.drivers.spads.dashboards.pyqtgraph import PyQtGraphDashboardConfig
from cc_hardware.tools.cli import register_cli, run_cli
from cc_hardware.utils import get_logger
from cc_hardware.utils.manager import Manager
from cc_hardware.utils.file_handlers import PklHandler

fx, fy = 615.71, 615.959
cx, cy = 321.125, 243.974

R = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])  # Rotate to make Z up
T = np.array([0, -2.5, -1.0])  # New origin 3m forward, 0.5m below
# T = np.array([0, 0, 0]) 
T_camera_to_global = np.eye(4)
T_camera_to_global[:3, :3] = R
T_camera_to_global[:3, 3] = T


def compute_3d_position(depth: np.ndarray, u: int, v: int):
    z = depth[v, u]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])


def to_global_position(
    camera_position: np.ndarray, T_camera_to_global: np.ndarray
) -> np.ndarray:
    # Convert to homogeneous coordinates
    camera_position_homogeneous = np.append(camera_position, 1)

    # Transform to global space
    global_position_homogeneous = T_camera_to_global @ camera_position_homogeneous
    return global_position_homogeneous[:3]


def filter_detections(result, name, confidence_threshold):
    # Filter by name and confidence
    filtered_boxes = []
    for i, box in enumerate(result.boxes):
        if result.names[int(box.cls)] == name and box.conf >= confidence_threshold:
            filtered_boxes.append(box)

    # Return filtered boxes as a list
    return filtered_boxes


fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(0, 5)


@register_cli
def camera_viewer(
    sensor: PklSPADSensorConfig = PklSPADSensorConfig(
        pkl_path="???", resolution=(3, 3), merge=False
    ),
    dashboard: SPADDashboardConfig = PyQtGraphDashboardConfig(fullscreen=False),
    output: Path | None = None,
    check_exists: bool = True,
    overwrite: bool = True,
    force: bool = False,
    iter: int = 0,
    remove_ambient: bool = False,
) -> bool | None:

    def setup(manager: Manager):
        _sensor: PklSPADSensor = sensor.create_instance(index=1)
        manager.add(sensor=_sensor)

        entry = _sensor.handler.load(0)
        if remove_ambient:
            if "ambient" in entry:
                manager.add(ambient=entry.get("ambient", None), primitive=True)
            elif "ambients" in entry:
                ambients = np.array(entry["ambients"])
                ambient = (np.mean(ambients, axis=0) / 2).astype(int)
                manager.add(ambient=ambient, primitive=True)
            else:
                raise ValueError("Ambient not found in the first frame.")
        else:
            manager.add(ambient=None)

        _output = sensor.pkl_path.with_name(
            output or f"{sensor.pkl_path.stem}_processed.pkl"
        )
        if _output.exists():
            if check_exists:
                get_logger().error(f"Output file {_output} already exists.")
                return False
            elif not force:
                input(
                    "Output file already exists. "
                    f"overwrite set to {overwrite}. Press Enter to continue..."
                )
        manager.add(handler=PklHandler(_output, overwrite=overwrite))

        _dashboard = dashboard.create_instance(sensor=_sensor)
        _dashboard.setup()
        manager.add(dashboard=_dashboard)

        manager.add(model=YOLO("yolo11n-seg"))


    def loop(
        iter: int,
        manager: Manager,
        sensor: PklSPADSensor,
        model: YOLO,
        dashboard: SPADDashboard,
        handler: PklHandler,
        ambient: np.ndarray | None,
    ) -> bool:
        get_logger().info(f"Capturing frame {iter}...")

        histogram, data = sensor.accumulate(return_entry=True, index=iter + 1)
        if ambient is not None and remove_ambient:
            assert histogram.shape == ambient.shape, "Ambient shape does not match."
            histogram -= ambient
            histogram = np.log1p(np.log1p(histogram))
            # normalize between 1 and 10000
            histogram = (histogram - np.min(histogram)) / (np.max(histogram) - np.min(histogram)) * 9999 + 1
            histogram = histogram.astype(int)
        dashboard.update(iter, histograms=histogram)

        rgb = data["rgb"]
        depth = data["depth"]

        result = model.predict(rgb, classes=[0], conf=0.4)[0]

        rgb_plot = result.plot()
        cv2.imshow("RGB", rgb_plot)
        waitKey = cv2.waitKey(1) & 0xFF
        # if waitKey == ord("q"):
        #     # Quit
        #     return False
        # elif waitKey == ord("n"):
        #     # Ignore this frame
        #     return True
        # elif waitKey == ord("y"):
        #     # Save this frame
        #     handler.append(data)
        #     return True
        # else:
        #     get_logger().warning("Invalid key pressed.")
        #     return False

        if waitKey == ord("q"):
            return False

        masks = result.masks
        if masks is not None and len(masks) != 0:
            if len(masks.data) > 1:
                masks.data = masks.data[0].unsqueeze(0)
            mask = masks.data.permute(1, 2, 0).numpy()

            # Color the uv points as red and paint the image
            mask = ops.scale_image(mask, rgb.shape[:2]).squeeze().astype(bool)
            uv = np.argwhere(mask)

            median_depth = np.median(depth[mask])
            median_uv = uv[np.argmin(np.abs(depth[uv[:, 0], uv[:, 1]] - median_depth))]
            position = compute_3d_position(depth, median_uv[1], median_uv[0]) / 1000
            position = to_global_position(position, T_camera_to_global)

            # Create blank image with position text
            rgb_position = rgb.copy()
            cv2.rectangle(rgb_position, (0, 0), (rgb_position.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(
                rgb_position,
                f"Median depth: {median_depth:.2f} mm",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.rectangle(rgb_position, (0, 50), (rgb_position.shape[1], 100), (0, 0, 0), -1)
            cv2.putText(
                rgb_position,
                f"Position: {position}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            cv2.imshow("RGB Position", rgb_position)
            cv2.waitKey(1)

            has_masks = True
            handler.append(
                dict(
                    position=position,
                    depth=depth,
                    median_depth=median_depth,
                    rgb=rgb,
                    masks=masks,
                    histogram=histogram,
                    has_masks=has_masks,
                )
            )

        else:
            has_masks = False
            handler.append(
                dict(
                    depth=depth,
                    rgb=rgb,
                    histogram=histogram,
                    has_masks=has_masks,
                )
            )

        return True

    with Manager() as manager:
        manager.run(iter, setup=setup, loop=loop)


if __name__ == "__main__":
    run_cli(camera_viewer)
