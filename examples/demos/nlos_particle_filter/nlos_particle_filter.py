import time
from datetime import datetime
from functools import partial
from pathlib import Path

from backprojection import (
    BackprojectionAlgorithm,
    BackprojectionConfig,
    BackprojectionDashboard,
    BackprojectionDashboardConfig,
)

from cc_hardware.drivers.spads import SPADDataType, SPADSensor, SPADSensorConfig
from cc_hardware.drivers.spads.spad_wrappers import SPADMovingAverageWrapperConfig
from cc_hardware.drivers.spads.vl53l8ch import VL53L8CHConfig8x8
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.tools.dashboard.spad_dashboard.pyqtgraph import (
    PyQtGraphDashboardConfig,
)
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers import PklHandler

# ==========

NOW = datetime.now()
LOGDIR: Path = Path("logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S")
OUTPUT_PKL: Path = LOGDIR / "data.pkl"

WRAPPED_SENSOR = VL53L8CHConfig8x8.create(
    num_bins=18,
    subsample=2,
    start_bin=35,
    integration_time_ms=100,
    data_type=SPADDataType.HISTOGRAM | SPADDataType.POINT_CLOUD | SPADDataType.DISTANCE,
)
SENSOR = SPADMovingAverageWrapperConfig.create(
    wrapped=WRAPPED_SENSOR,
    window_size=20,
)

DASHBOARD = PyQtGraphDashboardConfig.create(fullscreen=True)
DASHBOARD = None


# ==========


def setup(
    manager: Manager,
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig | None,
    record: bool = False,
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

    _sensor: SPADSensor = SPADSensor.create_from_config(sensor)
    manager.add(sensor=_sensor)

    if dashboard is not None:
        dashboard: SPADDashboard = dashboard.create_from_registry(
            config=dashboard, sensor=_sensor
        )
        dashboard.setup()
        manager.add(dashboard=dashboard)

    # Initialize the backprojection algorithm
    backprojection_config = BackprojectionConfig(
        x_range=(-0.5, 0.5),
        y_range=(-0.5, 0.5),
        z_range=(0, 1.5),
        num_x=50,
        num_y=50,
        num_z=50,
    )
    backprojection_algorithm = BackprojectionAlgorithm(
        backprojection_config, sensor_config=sensor
    )
    manager.add(algorithm=backprojection_algorithm)

    backprojection_dashboard_config = BackprojectionDashboardConfig(
        xlim=(-0.5, 0.5),
        ylim=(-0.5, 0.5),
        zlim=(0, 1),
        xres=0.01,
        yres=0.01,
        zres=0.01,
        num_x=50,
        num_y=50,
        num_z=50,
    )
    backprojection_dashboard = BackprojectionDashboard(backprojection_dashboard_config)
    manager.add(backprojection_dashboard=backprojection_dashboard)


import matplotlib.pyplot as plt
import numpy as np


def plot_axis(
    volume: np.ndarray,
    axis: str,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    xres: float,
    yres: float,
    zres: float,
    num_x: int,
    num_y: int,
    num_z: int,
    points: list[np.ndarray] | None = None,
    gt: list[np.ndarray] | None = None,
    idx: int | None = None,
    **kwargs,
) -> tuple[np.ndarray, float]:
    img: np.ndarray
    xnum, ynum = None, None
    val = None
    assert axis in ["x", "y", "z"], f"Invalid axis {axis}"
    if axis == "x":
        img = volume[idx, :, :].T if idx is not None else volume
        xlim, ylim = zlim, ylim
        xnum, ynum = num_z, num_y
        val = zlim[0] + (idx or 1) * zres
        xidx, yidx = 2, 1
    elif axis == "y":
        img = volume[:, idx, :].T if idx is not None else volume
        xlim, ylim = zlim, xlim
        xnum, ynum = num_z, num_x
        val = xlim[0] + (idx or 1) * xres
        xidx, yidx = 2, 0
    elif axis == "z":
        img = volume[:, :, idx].T if idx is not None else volume
        xlim, ylim = xlim, ylim
        xnum, ynum = num_x, num_y
        val = zlim[0] + (idx or 1) * zres
        xidx, yidx = 0, 1

    plt.imshow(img, **kwargs)

    xticks = np.round(np.linspace(0, xnum - 1, 5), 2)
    xlabels = np.round(np.linspace(xlim[0], xlim[1], 5), 2)
    plt.xticks(xticks, xlabels)

    yticks = np.round(np.linspace(0, ynum - 1, 5), 2)
    ylabels = np.round(np.linspace(ylim[0], ylim[1], 5), 2)
    plt.yticks(yticks, ylabels)

    points = [] if points is None else points
    for point in points:
        x = (point[xidx] - xlim[0]) / (xlim[1] - xlim[0]) * (num_x - 1)
        y = (point[yidx] - ylim[0]) / (ylim[1] - ylim[0]) * (num_y - 1)
        plt.plot(x, y, "og", markersize=10)

    gt = [] if gt is None else gt
    for gt_point in gt:
        x = (gt_point[xidx] - xlim[0]) / (xlim[1] - xlim[0]) * (num_x - 1)
        y = (gt_point[yidx] - ylim[0]) / (ylim[1] - ylim[0]) * (num_y - 1)
        plt.plot(x, y, "or", markersize=10)

    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    return img, val


def plot_volume_projection(
    volume: np.ndarray,
    title: str,
    gamma: int = 2,
    project_fn=np.max,
    fig: plt.Figure | None = None,
    signal: np.ndarray | None = None,
    **kwargs,
):
    import matplotlib.colors as mcolors

    kwargs.setdefault("cmap", "hot")
    kwargs.setdefault("norm", mcolors.PowerNorm(gamma=gamma))

    # normalize
    x_slice = project_fn(volume, axis=0)
    # x_slice = np.interp(x_slice, (x_slice.min(), x_slice.max()), (0, 1))
    y_slice = project_fn(volume, axis=1)
    # y_slice = np.interp(y_slice, (y_slice.min(), y_slice.max()), (0, 1))
    z_slice = project_fn(volume, axis=2)
    # z_slice = np.interp(z_slice, (z_slice.min(), z_slice.max()), (0, 1))

    num = 3 if signal is None else 4
    if fig is None:
        size = 4
        fig = plt.figure(figsize=(size * num, size))

    # Y-Z
    plt.subplot(1, num, 1)
    plot_axis(x_slice, "x", **kwargs)
    plt.title("Y-Z Projection")
    plt.xlabel("Z (m)")
    plt.ylabel("Y (m)")
    plt.gca().invert_xaxis()

    # X-Z
    plt.subplot(1, num, 2)
    plot_axis(y_slice, "y", **kwargs)
    plt.title("X-Z Projection")
    plt.ylabel("X (m)")
    plt.xlabel("Z (m)")
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    # X-Y
    plt.subplot(1, num, 3)
    plot_axis(z_slice.T, "z", **kwargs)
    plt.title("X-Y Projection")
    plt.ylabel("Y (m)")
    plt.xlabel("X (m)")

    # Signal
    if signal is not None:
        plt.subplot(1, num, 4)
        plt.imshow(signal, cmap="hot", aspect="auto")
        plt.title("Signal")

    plt.suptitle(title)

    plt.colorbar()


def loop(
    frame: int,
    manager: Manager,
    sensor: SPADSensor,
    dashboard: SPADDashboard | None = None,
    writer: PklHandler | None = None,
    algorithm: BackprojectionAlgorithm | None = None,
    backprojection_dashboard: BackprojectionDashboard | None = None,
):
    """Updates dashboard each frame.

    Args:
        frame (int): Current frame number.
        manager (Manager): Manager controlling the loop.
        sensor (SPADSensor): Sensor instance (unused here).
        dashboard (SPADDashboard): Dashboard instance to update.
    """
    global t0

    if frame % 10 == 0:
        t1 = time.time()
        fps = 10 / (t1 - t0)
        t0 = time.time()
        get_logger().info(f"Frame: {frame}, FPS: {fps:.2f}")

    data = sensor.accumulate()
    if dashboard is not None:
        dashboard.update(frame, data=data)

    if algorithm is not None:
        assert SPADDataType.POINT_CLOUD in data
        assert SPADDataType.HISTOGRAM in data

        volume = algorithm.update(data)

        # xy_slice = volume.mean(axis=2).T
        # plt.imshow(xy_slice, cmap="hot")

        # histogram = data[SPADDataType.HISTOGRAM]
        # # x_res = (x_range[1] - x_range[0]) / num_x
        # # y_res = (y_range[1] - y_range[0]) / num_y
        # # z_res = (z_range[1] - z_range[0]) / num_z
        # x_res = (algorithm.config.x_range[1] - algorithm.config.x_range[0]) / algorithm.config.num_x
        # y_res = (algorithm.config.y_range[1] - algorithm.config.y_range[0]) / algorithm.config.num_y
        # z_res = (algorithm.config.z_range[1] - algorithm.config.z_range[0]) / algorithm.config.num_z
        # plot_volume_projection(
        #     volume,
        #     title="Summed Volume Projection",
        #     # signal=histogram.mean(axis=(0, 1)),
        #     xlim=algorithm.config.x_range,
        #     ylim=algorithm.config.y_range,
        #     zlim=algorithm.config.z_range,
        #     num_x=algorithm.config.num_x,
        #     num_y=algorithm.config.num_y,
        #     num_z=algorithm.config.num_z,
        #     xres=x_res,
        #     yres=y_res,
        #     zres=z_res,
        #     # points=np.mean(object2origin_to_use[:, :3, 3], axis=0, keepdims=True),
        #     # gt=[best_summed_point]
        # )
        # plt.show()

        if backprojection_dashboard is not None:

            def filter_volume(volume: np.ndarray, num_x, num_y) -> np.ndarray:
                volume_unpadded = (
                    2 * volume[:, :, 1:-1] - volume[:, :, :-2] - volume[:, :, 2:]
                )
                zero_pad = np.zeros((num_x, num_y, 1))
                volume_padded = np.concatenate(
                    [zero_pad, volume_unpadded, zero_pad], axis=-1
                )
                return volume_padded

            filtered_volume = filter_volume(
                volume,
                num_x=algorithm.config.num_x,
                num_y=algorithm.config.num_y,
            )
            backprojection_dashboard.update(
                filtered_volume,
                data[SPADDataType.HISTOGRAM].reshape(
                    -1, data[SPADDataType.HISTOGRAM].shape[-1]
                ),
            )

    if writer is not None:
        writer.append({"iter": frame, **data})


@register_cli
def nlos_particle_filter_demo(record: bool = False):
    """Sets up and runs the SPAD dashboard.

    Args:
        sensor (SPADSensorConfig): Configuration object for the sensor.
        dashboard (SPADDashboardConfig): Configuration object for the dashboard.
    """

    global t0
    t0 = time.time()

    with Manager() as manager:
        manager.run(
            setup=partial(setup, record=record, sensor=SENSOR, dashboard=DASHBOARD),
            loop=loop,
        )


if __name__ == "__main__":
    run_cli(nlos_particle_filter_demo)
