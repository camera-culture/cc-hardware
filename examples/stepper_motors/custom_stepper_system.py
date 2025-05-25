import time
from dataclasses import field
from functools import partial

import numpy as np

from cc_hardware.drivers.stepper_motors import StepperMotorSystem
from cc_hardware.drivers.stepper_motors.stepper_controller import (
    ControllerAxisConfig,
    StepperController,
    StepperControllerConfig,
)
from cc_hardware.drivers.stepper_motors.stepper_system import StepperMotorSystemAxis
from cc_hardware.drivers.stepper_motors.telemetrix_stepper import (
    TelemetrixStepperMotorSystemConfig,
    TelemetrixStepperMotorXConfig,
)
from cc_hardware.utils import Config, Manager, config_wrapper, get_logger, run_cli

# ===============


@config_wrapper
class CustomLinearActuatorConfig(TelemetrixStepperMotorXConfig):
    cm_per_rev: float = 4
    steps_per_rev: int = 200
    speed: int = 500


@config_wrapper
class CustomStepperMotorSystemConfig(TelemetrixStepperMotorSystemConfig):
    axes: dict[StepperMotorSystemAxis, CustomLinearActuatorConfig] = field(
        default_factory=lambda: {StepperMotorSystemAxis.X: CustomLinearActuatorConfig}
    )


# ===============


@config_wrapper
class LinearControllerAxisConfig(Config):
    """Axis config for a linear stepper controller."""

    range: tuple[float, float]
    samples: int


@config_wrapper
class LinearStepperControllerConfig(StepperControllerConfig):
    """Config for a linear stepper controller."""

    axis: LinearControllerAxisConfig = field(
        default_factory=lambda: LinearControllerAxisConfig(
            range=(0.0, 10.0), samples=11
        )
    )
    bounce: bool = False


class LinearStepperController(StepperController[LinearStepperControllerConfig]):
    def __init__(self, config: LinearStepperControllerConfig):
        super().__init__(config)

        positions = np.linspace(
            config.axis.range[0], config.axis.range[1], config.axis.samples
        )
        self.axes = {
            "x": ControllerAxisConfig(index=0, positions=positions, flipped=False)
        }
        self.bounce = config.bounce

        if self.bounce:
            self.total_positions = len(positions) * 2 - 2  # ping‑pong
        else:
            self.total_positions = len(positions)

    def get_position(self, iter: int) -> dict | None:
        if iter >= self.total_positions:
            return None

        pos_list = self.axes["x"].positions
        if self.bounce and iter >= len(pos_list):
            idx = self.total_positions - iter
        else:
            idx = iter

        current = {"x": float(pos_list[idx])}
        get_logger().info(f"Axis x: {current['x']}")
        return current


# ===============


def setup(manager: Manager, port: str | None = None):
    controller = LinearStepperController(LinearStepperControllerConfig.create())
    manager.add(controller=controller)

    stepper_system = StepperMotorSystem.create_from_config(
        CustomStepperMotorSystemConfig.create(port=port)
    )
    stepper_system.initialize()
    manager.add(stepper_system=stepper_system)


def loop(
    iter: int,
    manager: Manager,
    controller: LinearStepperController,
    stepper_system: StepperMotorSystem,
    repeat: bool = True,
    **kwargs,
) -> bool:
    get_logger().info(f"Starting iter {iter}...")

    if repeat:
        iter = iter % controller.total_positions

    pos = controller.get_position(iter)
    if pos is None:
        return False

    stepper_system.move_to(pos["x"])
    time.sleep(1)


def custom_stepper_system(port: str | None = None, repeat: bool = True):
    with Manager() as manager:
        manager.run(setup=partial(setup, port=port), loop=partial(loop, repeat=repeat))


# ===============

if __name__ == "__main__":
    run_cli(custom_stepper_system)
