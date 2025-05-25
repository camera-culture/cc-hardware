import time
from functools import partial

from cc_hardware.drivers.stepper_motors import (
    StepperMotorSystem,
    StepperMotorSystemConfig,
)
from cc_hardware.drivers.stepper_motors.stepper_controller import (
    StepperController,
    StepperControllerConfig,
)
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli

# ===============


def setup(
    manager: Manager,
    stepper_system: StepperMotorSystemConfig,
    controller: StepperControllerConfig,
):
    _controller = StepperController.create_from_config(controller)
    manager.add(controller=_controller)

    _stepper_system = StepperMotorSystem.create_from_config(stepper_system)
    _stepper_system.initialize()
    manager.add(stepper_system=_stepper_system)


def loop(
    iter: int,
    manager: Manager,
    controller: StepperController,
    stepper_system: StepperMotorSystem,
    repeat: bool = True,
) -> bool:
    get_logger().info(f"Starting iter {iter}...")

    if repeat:
        iter = iter % controller.total_positions

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
def spad_dashboard_demo(
    stepper_system: StepperMotorSystemConfig,
    controller: StepperControllerConfig,
    repeat: bool = True,
):
    """Sets up and runs the stepper motor controller.

    Args:
        stepper_system (StepperMotorSystemConfig): Configuration for the stepper motor
            system.
        controller (StepperControllerConfig): Configuration for the stepper motor
            controller.
    """

    with Manager() as manager:
        manager.run(
            setup=partial(setup, stepper_system=stepper_system, controller=controller),
            loop=partial(loop, repeat=repeat),
            cleanup=cleanup,
        )


# ===============

if __name__ == "__main__":
    run_cli(spad_dashboard_demo)
