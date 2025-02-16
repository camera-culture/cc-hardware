import time

from cc_hardware.drivers.stepper_motors import StepperMotorSystem
from cc_hardware.drivers.stepper_motors.stepper_controller import (
    LinearStepperController,
)
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.manager import Manager
from cc_hardware.utils import register
from cc_hardware.drivers.stepper_motors.stepper_system import StepperMotorSystemAxis
from cc_hardware.drivers.stepper_motors.telemetrix_stepper import (
    TelemetrixStepperMotorSystem,
    TelemetrixStepperMotorX,
)

# ===============


@register
class CustomLinearActuator(TelemetrixStepperMotorX):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cm_per_rev", 4)
        kwargs.setdefault("steps_per_rev", 200)
        kwargs.setdefault("speed", 500)

        super().__init__(*args, **kwargs)


@register
class CustomStepperMotorSystem(TelemetrixStepperMotorSystem):
    def __init__(self, *args, **kwargs):
        axes = {StepperMotorSystemAxis.X: [CustomLinearActuator]}
        super().__init__(*args, axes=axes, **kwargs)


# ===============


STEPPER_SYSTEM_NAME: str = "CustomStepperMotorSystem"
STEPPER_PORT: str | None = None
CONTROLLER_CONFIG: dict = dict(name="x", range=(0, 16), samples=10)

# ===============


def setup(manager: Manager):
    controller = LinearStepperController(CONTROLLER_CONFIG)
    manager.add(controller=controller)

    stepper_system = StepperMotorSystem.create_from_registry(
        STEPPER_SYSTEM_NAME, port=STEPPER_PORT
    )
    stepper_system.initialize()
    manager.add(stepper_system=stepper_system)


def loop(
    iter: int,
    manager: Manager,
    controller: LinearStepperController,
    stepper_system: CustomLinearActuator,
    **kwargs,
) -> bool:
    get_logger().info(f"Starting iter {iter}...")

    pos = controller.get_position(iter)
    if pos is None:
        return False

    stepper_system.move_to(pos['x'])
    time.sleep(1)


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
