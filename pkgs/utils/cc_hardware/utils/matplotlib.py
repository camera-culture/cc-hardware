"""Plotting utilities for sensor data visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cc_hardware.utils.logger import get_logger


def set_matplotlib_style(*, use_scienceplots: bool = True):
    """Set the default style for matplotlib plots."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme("paper", font_scale=1.5)
    sns.set_style("ticks")

    plt.rcParams["figure.autolayout"] = True

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
