import time
from functools import partial

from cc_hardware.drivers.stepper_motors import (
    StepperMotorSystem,
    StepperMotorSystemConfig,
)
from cc_hardware.drivers.stepper_motors.stepper_controller import SnakeStepperController
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli

# ===============

CONTROLLER_CONFIG: list[dict] = [
    {"name": "x", "range": (0, 16), "samples": 10},
    {"name": "y", "range": (0, 16), "samples": 10},
]

# ===============


def setup(manager: Manager, stepper_system: StepperMotorSystemConfig):
    controller = SnakeStepperController(CONTROLLER_CONFIG)
    manager.add(controller=controller)

    _stepper_system = StepperMotorSystem.create_from_config(stepper_system)
    _stepper_system.initialize()
    manager.add(stepper_system=_stepper_system)


def loop(
    iter: int,
    manager: Manager,
    controller: SnakeStepperController,
    stepper_system: StepperMotorSystem,
) -> bool:
    get_logger().info(f"Starting iter {iter}...")

    pos = controller.get_position(iter)
    if pos is None:
        return False

    stepper_system.move_to(pos["x"], pos["y"])

    time.sleep(0.5)

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
def spad_dashboard_demo(stepper_system: StepperMotorSystemConfig):
    """Sets up and runs the stepper motor controller.

    Args:
        stepper_system (StepperMotorSystemConfig): Configuration for the stepper motor
            system.
    """

    with Manager() as manager:
        manager.run(
            setup=partial(setup, stepper_system=stepper_system),
            loop=loop,
            cleanup=cleanup,
        )


# ===============

if __name__ == "__main__":
    run_cli(spad_dashboard_demo)
