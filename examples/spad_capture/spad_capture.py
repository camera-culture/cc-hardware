from datetime import datetime
from pathlib import Path

from cc_hardware.drivers.spads import SPADSensor, SPADSensorConfig
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers import PklHandler

NOW = datetime.now()


@register_cli
def spad_dashboard(
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    save_data: bool = True,
    filename: Path | None = None,
    logdir: Path = Path("logs") / NOW.strftime("%Y-%m-%d"),
):
    """Sets up and runs the SPAD dashboard.

    Args:
        sensor (SPADSensorConfig): Configuration object for the sensor.
        dashboard (SPADDashboardConfig): Configuration object for the dashboard.
    """

    def setup(manager: Manager):
        if save_data:
            assert filename is not None, "Filename must be provided if saving data."
            assert (
                logdir / filename
            ).suffix == ".pkl", "Filename must have .pkl extension."
            assert not (
                logdir / filename
            ).exists(), "File already exists. Please provide a new filename."
            logdir.mkdir(exist_ok=True, parents=True)
            manager.add(writer=PklHandler(logdir / filename))

        _sensor: SPADSensor = SPADSensor.create_from_config(sensor)
        manager.add(sensor=_sensor)

        _dashboard: SPADDashboard = dashboard.create_from_registry(
            config=dashboard, sensor=_sensor
        )
        _dashboard.setup()
        manager.add(dashboard=_dashboard)

    def loop(
        frame: int,
        manager: Manager,
        sensor: SPADSensor,
        dashboard: SPADDashboard,
        writer: PklHandler | None = None,
    ):
        """Updates dashboard each frame.

        Args:
            frame (int): Current frame number.
            manager (Manager): Manager controlling the loop.
            sensor (SPADSensor): Sensor instance (unused here).
            dashboard (SPADDashboard): Dashboard instance to update.
        """
        get_logger().info(f"Starting iter {frame}...")

        histograms = sensor.accumulate()
        dashboard.update(frame, histograms=histograms)

        if save_data:
            assert writer is not None
            writer.append(
                {
                    "iter": iter,
                    "histogram": histograms,
                }
            )

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


if __name__ == "__main__":
    run_cli(spad_dashboard)
