import time
from datetime import datetime
from functools import partial
from pathlib import Path

from cc_hardware.drivers.spads import SPADDataType, SPADSensor, SPADSensorConfig
from cc_hardware.drivers.stepper_motors import (
    StepperMotorSystem,
    StepperMotorSystemConfig,
)
from cc_hardware.drivers.stepper_motors.stepper_controller import (
    StepperController,
    StepperControllerConfig,
)
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils import get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers import PklHandler
from cc_hardware.utils.manager import Manager

# ===============

# Uncomment to set the logger to use debug mode
# get_logger(level=logging.DEBUG)

# ===============

NOW = datetime.now()

# ===============


def setup(
    manager: Manager,
    *,
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    stepper_system: StepperMotorSystemConfig,
    controller: StepperControllerConfig,
    logdir: Path,
    record: bool = True,
):
    _sensor = SPADSensor.create_from_config(sensor)
    if not _sensor.is_okay:
        get_logger().fatal("Failed to initialize spad")
        return
    manager.add(sensor=_sensor)

    _dashboard = SPADDashboard.create_from_config(dashboard, sensor=_sensor)
    _dashboard.setup()
    manager.add(dashboard=_dashboard)

    controller = StepperController.create_from_config(controller)
    manager.add(controller=controller)

    _stepper_system = StepperMotorSystem.create_from_config(stepper_system)
    _stepper_system.initialize()
    manager.add(stepper_system=_stepper_system)

    if record:
        logdir.mkdir(parents=True, exist_ok=True)
        output_pkl = logdir / "data.pkl"
        assert not output_pkl.exists(), f"Output file {output_pkl} already exists"
        manager.add(writer=PklHandler(output_pkl))


def loop(
    iter: int,
    manager: Manager,
    sensor: SPADSensor,
    dashboard: SPADDashboard,
    controller: StepperController,
    stepper_system: StepperMotorSystem,
    writer: PklHandler | None = None,
    **kwargs,
) -> bool:
    get_logger().info(f"Starting iter {iter}...")

    data = sensor.accumulate()
    assert SPADDataType.HISTOGRAM in data, "Sensor must support histogram data type."
    dashboard.update(iter, data=data)

    pos = controller.get_position(iter)
    if pos is None:
        return False

    stepper_system.move_to(pos["x"], pos["y"])

    if writer is not None:
        writer.append(
            {
                "iter": iter,
                "pos": pos,
                "histogram": data[SPADDataType.HISTOGRAM],
            }
        )

    time.sleep(0.25)

    return True


def cleanup(
    stepper_system: StepperMotorSystem,
    **kwargs,
):
    get_logger().info("Cleaning up...")
    stepper_system.move_to(0, 0)
    stepper_system.close()


# ===============


@register_cli
def spad_stepper_system_capture(
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    stepper_system: StepperMotorSystemConfig,
    controller: StepperControllerConfig,
    logdir: Path = Path("logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S"),
    record: bool = True,
):
    _setup = partial(
        setup,
        sensor=sensor,
        dashboard=dashboard,
        stepper_system=stepper_system,
        controller=controller,
        logdir=logdir,
        record=record,
    )

    with Manager() as manager:
        manager.run(setup=_setup, loop=loop, cleanup=cleanup)


# ===============

if __name__ == "__main__":
    run_cli(spad_stepper_system_capture)
