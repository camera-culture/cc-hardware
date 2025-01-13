import time
from datetime import datetime
from functools import partial
from pathlib import Path

from cc_hardware.drivers.spads import SPADSensor
from cc_hardware.drivers.stepper_motors import StepperMotorSystem
from cc_hardware.drivers.stepper_motors.stepper_controller import StepperController
from cc_hardware.tools.dashboards import SPADDashboard
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
    gantry: StepperMotorSystemConfig,
    controller: StepperControllerConfig,
    logdir: Path,
):
    logdir.mkdir(parents=True, exist_ok=True)

    spad = SPADSensor.create_from_config(sensor)
    if not spad.is_okay:
        get_logger().fatal("Failed to initialize spad")
        return
    manager.add(spad=spad)

    dashboard = SPADDashboard.create_from_config(dashboard, sensor=spad)
    dashboard.setup()
    manager.add(dashboard=dashboard)

    controller = StepperController.create_from_config(controller)
    manager.add(controller=controller)

    gantry = StepperMotorSystem.create_from_config(gantry)
    gantry.initialize()
    manager.add(gantry=gantry)

    output_pkl = logdir / "data.pkl"
    assert not output_pkl.exists(), f"Output file {output_pkl} already exists"
    manager.add(writer=PklHandler(output_pkl))


def loop(
    iter: int,
    manager: Manager,
    spad: SPADSensor,
    dashboard: SPADDashboard,
    controller: StepperController,
    stepper_system: StepperMotorSystem,
    writer: PklHandler,
    **kwargs,
) -> bool:
    get_logger().info(f"Starting iter {iter}...")

    histogram = spad.accumulate()
    dashboard.update(iter, histograms=histogram)

    pos = controller.get_position(iter)
    if pos is None:
        return False

    stepper_system.move_to(pos["x"], pos["y"])

    writer.append(
        {
            "iter": iter,
            "pos": pos,
            "histogram": histogram,
        }
    )

    time.sleep(0.25)

    return True


# ===============


@register_cli
def spad_gantry_capture_v2(
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    gantry: StepperMotorSystemConfig,
    controller: StepperControllerConfig,
    logdir: Path = Path("logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S"),
):
    _setup = partial(
        setup,
        sensor=sensor,
        dashboard=dashboard,
        gantry=gantry,
        controller=controller,
        logdir=logdir,
    )

    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


# ===============

if __name__ == "__main__":
    run_cli(spad_gantry_capture_v2)
