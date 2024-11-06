from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from cc_hardware.drivers.spad import SPADSensor
from cc_hardware.utils.logger import get_logger


def set_matplotlib_style(*, use_scienceplots: bool = True):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme("paper", font_scale=1.5)
    sns.set_style("ticks")

    if use_scienceplots:
        from matplotlib import rcParams

        styles = ["science", "nature"]
        if not rcParams.get("tex.usetex", False):
            styles += ["no-latex"]

        try:
            import scienceplots  # noqa

            plt.style.use(styles)
        except ImportError:
            get_logger().warning(
                "SciencePlots not found. Using default matplotlib style."
            )


def histogram_gui(
    sensor: SPADSensor,
    *,
    num_frames: int = 100,
    show: bool = True,
    save: bool = False,
    filename: str = None,
    min_bin: int = 0,
    max_bin: int = 128,
    autoscale: bool = True,
    ylim: float | None = 5000,
    channel_mask: list[int] | None = None,
):
    assert save or show, "Either show or save must be True."
    assert not save or filename, "Filename must be provided if save is True."

    get_logger().info("Starting histogram GUI...")

    set_matplotlib_style()

    h, w = sensor.resolution
    total_channels = h * w
    if channel_mask is None:
        channel_mask = np.arange(total_channels)  # Show all channels if no mask
    channel_mask = np.array(channel_mask)
    num_channels = len(channel_mask)

    # Determine grid size using numpy
    rows = int(np.ceil(np.sqrt(num_channels)))
    cols = int(np.ceil(num_channels / rows))

    fig = plt.figure(figsize=(10, 10 * rows / cols))
    gs = plt.GridSpec(rows, cols, figure=fig)

    # Setup
    bins = np.arange(min_bin, max_bin)
    colors = plt.cm.viridis(np.linspace(0, 1, num_channels))

    # Initialize each subplot
    axes = []
    containers = []
    for idx, channel in enumerate(channel_mask):
        row, col = divmod(idx, cols)
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

        container = ax.hist(
            bins,
            bins=bins.size,
            weights=np.zeros(bins.size),
            edgecolor="black",
            color=colors[idx],
        )[2]
        containers.append(container)

        ax.set_xlim(min_bin, max_bin)
        ax.set_xlabel("Bin")
        ax.set_ylabel("Photon Counts")

    # Update function
    def update(frame: int) -> list:
        if not plt.fignum_exists(fig.number) and show:
            get_logger().info("Closing GUI...")
            return

        # get_logger().info(f"Frame {frame}")

        histograms = sensor.accumulate(1)

        for idx, channel in enumerate(channel_mask):
            container = containers[idx]
            histogram = histograms[channel, min_bin:max_bin]

            for rect, h in zip(container, histogram):
                rect.set_height(h)

        # Update y-axis limits
        if ylim is not None:
            for ax in axes:
                ax.set_ylim(0, ylim)
        elif autoscale:
            for ax, idx in zip(axes, channel_mask):
                histogram = histograms[idx, min_bin:max_bin]
                ax.set_ylim(0, histogram.max())
        else:
            # Set all y-axis limits to the same value
            max_count = np.max(
                [histograms[channel, min_bin:max_bin].max() for channel in channel_mask]
            )
            for ax in axes:
                ax.set_ylim(0, max_count)

        plt.tight_layout()

        return list(chain(*containers))

    # Create animation
    ani = FuncAnimation(
        fig,
        update,
        frames=range(num_frames),
        interval=1,
        repeat=False,
        blit=True,
    )

    # Show the GUI
    if show:
        plt.show()

    # Save the animation as mp4 if required
    if save and filename:
        if not filename.endswith(".mp4"):
            filename += ".mp4"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        get_logger().info(f"Saving animation to {filename}...")
        ani.save(filename, writer="ffmpeg", dpi=400, fps=10)
        plt.close(fig)


def plot_points(
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    x_pred: np.ndarray,
    y_pred: np.ndarray,
    check_unique: bool = False,
    fig: plt.Figure | None = None,
    filename: Path | str | None = None,
):
    if check_unique:
        # Find unique actual positions and compute the mean of the corresponding
        # predicted values
        gt_points = np.vstack([x_pos, y_pos]).T
        unique_gt, indices = np.unique(gt_points, axis=0, return_index=True)
        inv_gt, inv_ind = np.unique(gt_points, axis=0, return_inverse=True)

        x_pred_means = np.array(
            [np.median(x_pred[inv_ind == i]) for i in range(len(inv_gt))]
        )
        y_pred_means = np.array(
            [np.median(y_pred[inv_ind == i]) for i in range(len(inv_gt))]
        )

        x_pos, y_pos = unique_gt[:, 0], unique_gt[:, 1]
        x_pred, y_pred = x_pred_means, y_pred_means

        # Reorder the points in the same order as they first appear in the gt data
        x_pos = x_pos[np.argsort(indices)]
        y_pos = y_pos[np.argsort(indices)]
        x_pred = x_pred[np.argsort(indices)]
        y_pred = y_pred[np.argsort(indices)]

    # First plot: Scatter plot for X and Y positions with correspondence lines and
    # movement path
    if fig is None:
        plt.figure(figsize=(5, 5))
    plt.scatter(x_pos, y_pos, label="Actual", color="blue")
    plt.scatter(x_pred, y_pred, label="Predicted", color="orange")

    # Draw lines to show correspondence between predicted and actual points
    for x1, y1, x2, y2 in zip(x_pos, y_pos, x_pred, y_pred):
        plt.plot([x1, x2], [y1, y2], "k--", alpha=0.5)

    # Draw movement path for ground truth positions
    for i in range(1, len(x_pos)):
        plt.arrow(
            x_pos[i - 1],
            y_pos[i - 1],
            x_pos[i] - x_pos[i - 1],
            y_pos[i] - y_pos[i - 1],
            head_width=0.1,
            head_length=0.2,
            fc="blue",
            ec="blue",
            alpha=0.7,
        )

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend(loc="upper right")
    if filename is not None:
        plt.savefig(filename)
    if fig is None:
        plt.close()
