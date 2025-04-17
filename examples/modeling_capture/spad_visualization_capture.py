from datetime import datetime
from pathlib import Path

from cc_hardware.drivers.spads import SPADSensor, SPADSensorConfig
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers import PklHandler

NOW = datetime.now()
LOGDIR: Path = Path("logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S")
OUTPUT_PKL: Path = LOGDIR / "data.pkl"

@register_cli
def capture_dashboard(
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    save_data: bool = True,
    sensor_port: str | None = None,
):
    """Sets up and runs the SPAD dashboard.

    Args:
        sensor (SPADSensorConfig): Configuration object for the sensor.
        dashboard (SPADDashboardConfig): Configuration object for the dashboard.
    """

    def setup(manager: Manager):
        if save_data:
            LOGDIR.mkdir(exist_ok=True, parents=True)

            OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
            assert not OUTPUT_PKL.exists(), f"Output file {OUTPUT_PKL} already exists"
            manager.add(writer=PklHandler(OUTPUT_PKL))
        
        sensor.port=sensor_port
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

# Example usage:
# python spad_visualization_capture.py sensor=VL53L8CHConfig4x4 dashboard=PyQtGraphDashboardConfig save_data=False sensor.integration_time_ms=100 sensor.cnh_num_bins=16 sensor.cnh_subsample=2 sensor.cnh_start_bin=20

if __name__ == "__main__":
    run_cli(capture_dashboard)