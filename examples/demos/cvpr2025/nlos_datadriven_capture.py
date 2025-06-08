import time
from datetime import datetime
from functools import partial
from pathlib import Path

import tqdm

from cc_hardware.drivers.spads import SPADDataType, SPADSensor, SPADSensorConfig
from cc_hardware.drivers.spads.spad_wrappers import SPADMovingAverageWrapperConfig
from cc_hardware.drivers.spads.vl53l8ch import RangingMode, VL53L8CHConfig8x8
from cc_hardware.drivers.stepper_motors import (
    StepperControllerConfig,
    StepperMotorSystem,
    StepperMotorSystemConfig,
)
from cc_hardware.drivers.stepper_motors.stepper_controller import (
    SnakeControllerAxisConfig,
    SnakeStepperControllerConfigXY,
    StepperController,
    StepperControllerConfig,
)
from cc_hardware.drivers.stepper_motors.telemetrix_stepper import (
    SingleDrive1AxisGantryConfig,
)
from cc_hardware.tools.dashboard.spad_dashboard import (
    DummySPADDashboardConfig,
    SPADDashboard,
    SPADDashboardConfig,
)
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
    start_bin=20,
    ranging_mode=RangingMode.CONTINUOUS,
    ranging_frequency_hz=9,
    data_type=SPADDataType.HISTOGRAM | SPADDataType.POINT_CLOUD | SPADDataType.DISTANCE,
)
# WRAPPED_SENSOR = SPADBackgroundRemovalWrapperConfig.create(
#     pkl_spad=PklSPADSensorConfig.create(
#         pkl_path=Path("logs") / "2025-06-05/11-26-31/data.pkl",
#         index=1,
#     ),
#     wrapped=WRAPPED_SENSOR,
# )
WRAPPED_SENSOR = SPADMovingAverageWrapperConfig.create(
    wrapped=WRAPPED_SENSOR,
    window_size=1,
)
SENSOR = WRAPPED_SENSOR

DASHBOARD = PyQtGraphDashboardConfig.create(fullscreen=True)
DASHBOARD = DummySPADDashboardConfig.create()

STEPPER_SYSTEM = SingleDrive1AxisGantryConfig.create()
STEPPER_CONTROLLER = SnakeStepperControllerConfigXY.create(
    axes=dict(
        x=SnakeControllerAxisConfig(range=(0, 32), samples=20),
        y=SnakeControllerAxisConfig(range=(0, 32), samples=20),
    )
)

REPETITIONS = 2
REDUNDANT_SAMPLES = 10

# ==========


def setup(
    manager: Manager,
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    stepper_system: StepperMotorSystemConfig,
    controller: StepperControllerConfig,
    record: bool = False,
    sensor_port: str | None = None,
    stepper_port: str | None = None,
    background: bool = True,
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

    _sensor: SPADSensor = SPADSensor.create_from_config(sensor, port=sensor_port)
    manager.add(sensor=_sensor)

    _dashboard: SPADDashboard = dashboard.create_from_registry(
        config=dashboard, sensor=_sensor
    )
    _dashboard.setup()
    manager.add(dashboard=_dashboard)

    if stepper_port is not None:
        stepper_system.port = stepper_port
    _stepper_system = StepperMotorSystem.create_from_config(stepper_system)
    _stepper_system.initialize()
    manager.add(stepper_system=_stepper_system)

    _controller = StepperController.create_from_config(controller)
    manager.add(controller=_controller)

    pbar = tqdm.tqdm(
        total=_controller.total_positions * REDUNDANT_SAMPLES * REPETITIONS,
        desc="Frames...",
        leave=False,
    )
    manager.add(pbar=pbar)

    # If background, capture N frames with frame = -1
    if background:
        input("Press Enter to start background capture...")
        for _ in range(REDUNDANT_SAMPLES):
            data = _sensor.accumulate()
            if writer is not None:
                writer.append({"iter": -1, **data})
        input("Background capture complete. Press Enter to continue...")

    LOGDIR.mkdir(exist_ok=True, parents=True)
    PklHandler(LOGDIR / "config.pkl").write(
        dict(
            sensor=sensor,
            dashboard=dashboard,
            stepper_system=stepper_system,
            controller=controller,
            pkl_path=OUTPUT_PKL,
        )
    )


def loop(
    frame: int,
    manager: Manager,
    sensor: SPADSensor,
    dashboard: SPADDashboard,
    stepper_system: StepperMotorSystem,
    controller: StepperController,
    pbar: tqdm.tqdm,
    writer: PklHandler | None = None,
):
    pbar.update(1)

    # how many REDUNDANT_SAMPLES‐blocks have we done?
    block = frame // REDUNDANT_SAMPLES
    rep = block // controller.total_positions
    if rep >= REPETITIONS:
        return False

    pos_idx = block % controller.total_positions

    # move & reset at the start of each block
    if frame % REDUNDANT_SAMPLES == 0:
        sensor.reset()
        pos = controller.get_position(pos_idx)
        stepper_system.move_to(pos["x"], pos["y"])

    data = sensor.accumulate()
    dashboard.update(frame, data=data)

    if writer is not None:
        writer.append(
            {
                "iter": frame,
                "pos": controller.get_position(pos_idx, verbose=False),
                **data,
            }
        )


def cleanup(
    stepper_system: StepperMotorSystem,
    **kwargs,
):
    get_logger().info("Cleaning up...")
    stepper_system.move_to(0, 0)
    stepper_system.close()


@register_cli
def nlos_datadriven_capture(
    sensor_port: str | None = None,
    stepper_port: str | None = None,
    record: bool = False,
    background: bool = True,
):
    """Sets up and runs the SPAD dashboard.

    Args:
        sensor (SPADSensorConfig): Configuration object for the sensor.
        dashboard (SPADDashboardConfig): Configuration object for the dashboard.
    """

    global t0
    t0 = time.time()

    with Manager() as manager:
        manager.run(
            setup=partial(
                setup,
                sensor_port=sensor_port,
                stepper_port=stepper_port,
                record=record,
                sensor=SENSOR,
                dashboard=DASHBOARD,
                stepper_system=STEPPER_SYSTEM,
                controller=STEPPER_CONTROLLER,
                background=background,
            ),
            loop=loop,
            cleanup=cleanup,
        )


if __name__ == "__main__":
    run_cli(nlos_datadriven_capture)
