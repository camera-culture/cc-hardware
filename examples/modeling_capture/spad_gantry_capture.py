from datetime import datetime
from pathlib import Path
import time

from cc_hardware.drivers.spads import SPADSensor, SPADSensorConfig
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers import PklHandler
from cc_hardware.drivers.stepper_motors import StepperMotorSystem
from cc_hardware.drivers.stepper_motors.stepper_controller import SnakeStepperController

import numpy as np

NOW = datetime.now()
LOGDIR: Path = Path("logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S")
OUTPUT_PKL: Path = LOGDIR / "data.pkl"

STEPPER_SYSTEM_NAME: str = "SingleDrive1AxisGantry"
STEPPER_PORT: str | None = None
CONTROLLER_CONFIG: list[dict] = [
    {"name": "x", "range": (0, 35), "samples": 10},
    {"name": "y", "range": (0, 42), "samples": 10},
]

NUM_CAPTURES_PER_ZONE: int = 5


@register_cli
def capture_dashboard(
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    save_data: bool = True,
    sensor_port: str | None = None,
    gantry_port: str | None = None,
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

        controller = SnakeStepperController(CONTROLLER_CONFIG)
        manager.add(controller=controller)

        STEPPER_PORT = gantry_port
        stepper_system = StepperMotorSystem.create_from_registry(
            STEPPER_SYSTEM_NAME, port=STEPPER_PORT, axes_kwargs={}
        )
        stepper_system.initialize()
        manager.add(stepper_system=stepper_system)

    def loop(
        frame: int,
        manager: Manager,
        sensor: SPADSensor,
        dashboard: SPADDashboard,
        controller: SnakeStepperController,
        stepper_system: StepperMotorSystem,
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

        pos = controller.get_position(frame)
        if pos is None:
            stepper_system.move_by(-10, 0)  # more than 35 to account for drift
            stepper_system.move_by(-10, 0)
            stepper_system.move_by(-10, 0)
            stepper_system.move_by(-5, 0)
            return False

        print(f"moving to: {pos}")
        stepper_system.move_to(pos["x"], pos["y"])

        time.sleep(0.5)

        histograms = sensor.accumulate(NUM_CAPTURES_PER_ZONE, average=False)
        if NUM_CAPTURES_PER_ZONE > 1:
            hist = np.mean(histograms, axis=0)
        else:
            hist = histograms
        dashboard.update(frame, histograms=hist)

        if save_data:
            assert writer is not None
            writer.append(
                {
                    "iter": frame,
                    "position": pos,
                    "histogram": histograms,
                }
            )

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


# Example usage:
# python spad_gantry_capture.py sensor=VL53L8CHConfig8x8 dashboard=PyQtGraphDashboardConfig sensor.integration_time_ms=100 sensor.cnh_num_bins=16 sensor.cnh_subsample=2 sensor.cnh_start_bin=0

if __name__ == "__main__":
    run_cli(capture_dashboard)