from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ultralytics.engine.results import Masks
from ultralytics.utils import ops

from cc_hardware.drivers.spads.dashboards.matplotlib import MatplotlibDashboard
from cc_hardware.drivers.spads.pkl import PklSPADSensor, PklSPADSensorConfig
from cc_hardware.tools.cli import register_cli, run_cli
from cc_hardware.utils import get_logger
from cc_hardware.utils.file_handlers import PklHandler
from cc_hardware.utils.manager import Manager


def print(*args):
    for arg in args:
        get_logger().info(arg)


fx, fy = 615.71, 615.959
cx, cy = 321.125, 243.974

R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # Rotate to make Z up
T = np.array([0, 2.75, -0.5])  # New origin 3m forward, 0.5m below
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


def format_array(array: np.ndarray) -> str:
    return np.array2string(array, formatter={"float_kind": lambda x: f"{x:.2f}"})


@register_cli(simple=True)
def algo_nlos_tmf8828_realsense(pkl_file: Path, output_file: Path | None = None):
    if output_file is None:
        output_file = pkl_file.with_name(pkl_file.stem + "_processed.pkl")

    def setup(manager: Manager):
        sensor = PklSPADSensor(
            PklSPADSensorConfig(pkl_path=pkl_file, resolution=(3, 3))
        )
        manager.add(sensor=sensor)

        manager.add(pbar=tqdm(len(sensor.handler), desc="Frames", leave=False))

        # dashboard = MatplotlibDashboard(MatplotlibDashboardConfig(ylim=1000), sensor)
        # dashboard.setup()
        # manager.add(dashboard=dashboard)
        manager.add(dashboard=None)

        # fig, ax = plt.subplots(1, 1)
        # manager.add(fig=fig, ax=ax)
        manager.add(fig=None, ax=None)

        manager.add(writer=PklHandler(output_file))

        _, data = manager["sensor"].accumulate(return_entry=True, index=0)
        print(data.keys())
        exit()

    def loop(
        iter: int,
        manager: Manager,
        sensor: PklSPADSensor,
        pbar: tqdm,
        dashboard: MatplotlibDashboard,
        writer: PklHandler,
        fig: plt.Figure,
        ax: plt.Axes,
    ):
        histogram, data = sensor.accumulate(return_entry=True, index=iter)

        person_depth: float = data["person_depth"]
        # histogram: np.ndarray = data["histogram"]
        rgb: np.ndarray = data["rgb"]
        depth: np.ndarray = data["depth"]
        masks: Masks = data["masks"]
        if masks.data is None:
            return True
        mask = masks.data.permute(1, 2, 0).numpy()

        # Color the uv points as red and paint the image
        mask = ops.scale_image(mask, rgb.shape[:2]).squeeze().astype(bool)
        uv = np.argwhere(mask)

        median_depth = np.median(depth[mask])
        median_uv = uv[np.argmin(np.abs(depth[uv[:, 0], uv[:, 1]] - median_depth))]
        position = compute_3d_position(depth, median_uv[1], median_uv[0]) / 1000
        position = to_global_position(position, T_camera_to_global)

        # # Draw bounding box with text label saying median depth
        # cv2.rectangle(rgb, (0, 0), (rgb.shape[1], 50), (0, 0, 0), -1)
        # cv2.putText(rgb, f"Median depth: {median_depth:.2f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # # Now put text of position
        # cv2.rectangle(rgb, (0, 50), (rgb.shape[1], 100), (0, 0, 0), -1)
        # cv2.putText(rgb, f"Position: {format_array(position)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if dashboard is not None:
            dashboard.update(iter, histograms=histogram)

        # cv2.imshow("RGB", rgb)
        # if cv2.waitKey(100) & 0xFF == ord("q"):
        #     return False

        writer.append({"position": position, "histogram": histogram})

        pbar.update(1)

        return True

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


if __name__ == "__main__":
    run_cli(algo_nlos_tmf8828_realsense)
