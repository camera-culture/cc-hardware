import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from cc_hardware.drivers.spad import SPADSensor
from cc_hardware.tools.app import APP
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.plotting import set_matplotlib_style


def histogram_gui(sensor: SPADSensor):
    get_logger().info("Starting histogram GUI...")

    set_matplotlib_style()

    h, w = sensor.resolution
    fig = plt.figure(figsize=(3 * w, 3 * h))
    gs = GridSpec(h, w, height_ratios=[1, 1, 1], figure=fig)

    # Setup
    bins = np.arange(0, sensor.num_bins)
    colors = plt.cm.viridis(np.linspace(0, 1, h * w))

    # Initialize each subplot
    axes = []
    containers = []
    for i in range(h):
        for j in range(w):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)

            container = ax.hist(
                bins,
                bins=bins.size,
                weights=np.zeros(bins.size),
                edgecolor="black",
                color=colors[i * w + j],
            )[2]
            containers.append(container)

            ax.set_xlim(bins.min(), bins.max())
            ax.set_xlabel("Bin")
            ax.set_ylabel("Photon Counts")

    # Update function
    def update(frame):
        if not plt.fignum_exists(fig.number):
            get_logger().info("Closing GUI...")
            return

        get_logger().info(f"Frame {frame}")

        histograms = sensor.accumulate(1)
        h, w = sensor.resolution

        for i in range(h * w):
            container = containers[i]
            histogram = histograms[i]

            for rect, h in zip(container, histogram):
                rect.set_height(h)

        # Update y-axis limits
        max_count = max([histogram.max() for histogram in histograms])
        for ax in axes:
            ax.set_ylim(0, max_count)

        plt.tight_layout()
        plt.pause(0.001)

    FuncAnimation(fig, update, frames=range(100), repeat=False)
    plt.show()


@APP.command()
def tmf8828_dashboard():
    from cc_hardware.drivers.spads.tmf8828 import TMF8828Sensor
    from cc_hardware.utils.manager import Manager

    def setup(manager: Manager, spad: TMF8828Sensor):
        histogram_gui(spad)

    with Manager(spad=TMF8828Sensor) as manager:
        manager.run(setup=setup)
