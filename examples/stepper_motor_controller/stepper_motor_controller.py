import time
from datetime import datetime
from pathlib import Path

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

STEPPER_SYSTEM_NAME: str = "SingleDrive1AxisGantry"
STEPPER_PORT: str | None = None
CONTROLLER_CONFIG: list[dict] = [
    {"name": "x", "range": (0, 16), "samples": 10},
    {"name": "y", "range": (0, 16), "samples": 10},
]

OUTPUT_PKL: Path = LOGDIR / "data.pkl"

# ===============


def setup(manager: Manager):
    LOGDIR.mkdir(parents=True, exist_ok=True)

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
    controller: SnakeStepperController,
    stepper_system: StepperMotorSystem,
    writer: PklHandler,
    **kwargs,
) -> bool:
    get_logger().info(f"Starting iter {iter}...")

    pos = controller.get_position(iter)
    if pos is None:
        return False

    stepper_system.move_to(pos["x"], pos["y"])

    writer.append(
        {
            "iter": iter,
            "pos": pos,
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

    parser = argparse.ArgumentParser(description="Demo: Stepper Motor Controller")

    parser.add_argument(
        "--port", default=None, help="The port to use for the stepper motor system."
    )

    args = parser.parse_args()

    STEPPER_PORT = args.port

    main()
