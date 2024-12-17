import time
from datetime import datetime
from pathlib import Path

from cc_hardware.drivers.spads import SPADSensor
from cc_hardware.drivers.spads.dashboard import MatplotlibDashboard
from cc_hardware.drivers.stepper_motors import StepperMotorSystem
from cc_hardware.drivers.stepper_motors.stepper_controller import SnakeStepperController
from cc_hardware.utils.file_handlers import PklHandler
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.manager import Manager

# ===============

# Uncomment to set the logger to use debug mode
# get_logger(level=logging.DEBUG)

# ===============

now = datetime.now()
LOGDIR: Path = Path("logs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")

SPAD_NAME: str = "VL53L8CHSensor"
SPAD_PORT: str

STEPPER_SYSTEM_NAME: str = "SingleDrive1AxisGantry"
STEPPER_PORT: str
CONTROLLER_CONFIG: list[dict] = [
    {"name": "x", "range": (0, 16), "samples": 10},
    {"name": "y", "range": (0, 16), "samples": 10},
]

OUTPUT_PKL: Path = LOGDIR / "data.pkl"

# ===============


def setup(manager: Manager):
    LOGDIR.mkdir(parents=True, exist_ok=True)

    spad = SPADSensor.create_from_registry(SPAD_NAME, port=SPAD_PORT)
    if not spad.is_okay:
        get_logger().fatal("Failed to initialize spad")
        return
    manager.add(spad=spad)

    dashboard = MatplotlibDashboard(spad)
    dashboard.setup()
    manager.add(dashboard=dashboard)

    controller = SnakeStepperController(CONTROLLER_CONFIG)
    manager.add(controller=controller)

    stepper_system = StepperMotorSystem.create_from_registry(
        STEPPER_SYSTEM_NAME, port=STEPPER_PORT
    )
    stepper_system.initialize()
    manager.add(stepper_system=stepper_system)

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    assert not OUTPUT_PKL.exists(), f"Output file {OUTPUT_PKL} already exists"
    manager.add(writer=PklHandler(OUTPUT_PKL))


def loop(
    iter: int,
    manager: Manager,
    spad: SPADSensor,
    dashboard: MatplotlibDashboard,
    controller: SnakeStepperController,
    stepper_system: StepperMotorSystem,
    writer: PklHandler,
    **kwargs,
) -> bool:
    get_logger().info(f"Starting iter {iter}...")

    histogram = spad.accumulate()
    dashboard.update(iter)

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


def main():
    with Manager() as manager:
        manager.run(setup=setup, loop=loop)


# ===============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo: SPAD and Gantry")

    parser.add_argument(
        "--spad-port", help="Port that the SPAD sensor is on.", required=True
    )
    parser.add_argument(
        "--gantry-port", help="Port that the gantry is on.", required=True
    )

    args = parser.parse_args()

    SPAD_PORT = args.spad_port
    STEPPER_PORT = args.gantry_port

    main()
