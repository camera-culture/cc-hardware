"""This module defines a system of stepper motors that can be controlled as a single
unit. This is useful for controlling a gantry or a multi-axis rotation stage."""

from enum import Enum
from typing import Any, Callable, overload

from cc_hardware.drivers.stepper_motors.stepper_motor import (
    DummyStepperMotor,
    StepperMotor,
    StepperMotorConfig,
)
from cc_hardware.utils import (
    Component,
    Config,
    call_async_gather,
    config_wrapper,
    get_logger,
)

# ======================


class StepperMotorSystemAxis(Enum):
    X = "X"
    Y = "Y"
    Z = "Z"

    ROLL = "ROLL"
    PITCH = "PITCH"
    YAW = "YAW"

    AZIMUTH = "AZIMUTH"
    ELEVATION = "ELEVATION"


@config_wrapper
class StepperMotorSystemConfig(Config):
    """Configuration for a stepper motor system.

    Attributes:
        axes (dict[StepperMotorSystemAxis, list[StepperMotor]]): A dictionary of axes
            and the motors that are attached to them.

    """

    axes: dict[StepperMotorSystemAxis, list[StepperMotorConfig]]


# ======================


class StepperMotorSystem[T: StepperMotorSystemConfig](Component[T]):
    """This is a wrapper around multiple stepper motors which defines the system
    as a whole (i.e. a gantry or multi-axis rotation stage).
    """

    def __init__(
        self,
        config: T,
    ):
        super().__init__(config)

        self._axes = {
            axis: [StepperMotor.create_from_config(motor) for motor in motors]
            for axis, motors in config.axes.items()
        }

    def initialize(self):
        get_logger().info(f"Initialized {self.__class__.__name__}.")

    @overload
    def move_to(self, *positions: float):
        ...

    @overload
    def move_to(self, **positions: float):
        ...

    def move_to(self, *args: float, **kwargs: float):
        """Move to the specified position using positional or keyword arguments."""
        if args and kwargs:
            raise ValueError("move_to takes either all positional or all keyword args.")
        elif args:
            assert len(args) == len(
                self.config.axes
            ), f"Got {len(args)} positions, expected {len(self.axes)}"
            positions = {axis: position for axis, position in zip(self.axes, args)}
        elif kwargs:
            assert len(kwargs) == len(
                self.axes
            ), f"Got {len(kwargs)} positions, expected {len(self.axes)}"
            positions = kwargs

        current_positions = self.position
        relative_positions = {
            axis.value: pos - current_pos
            for (axis, pos), current_pos in zip(positions.items(), current_positions)
        }
        self.move_by(**relative_positions)

    @overload
    def move_by(self, *positions: float):
        ...

    @overload
    def move_by(self, block: bool = True, **positions: float):
        ...

    def move_by(self, *args: float, **kwargs: float):
        """Moves the steppers to the specified positions."""
        if args and kwargs:
            raise ValueError("move_to takes either all positional or all keyword args.")
        elif args:
            assert len(args) == len(
                self.axes
            ), f"Got {len(args)} args, expected {len(self.axes)}"
            positions = {axis: position for axis, position in zip(self.axes, args)}
        elif kwargs:
            assert len(kwargs) == len(
                self.axes
            ), f"Got {len(kwargs)} kwargs, expected {len(self.axes)}"
            positions = {
                StepperMotorSystemAxis[axis.upper()]: pos
                for axis, pos in kwargs.items()
            }

        # Set the target position of each motor
        for axis, position in positions.items():
            for motor in self.axes[axis]:
                get_logger().info(f"Moving {axis} by {position}...")
                motor.move_by(position)

        self.wait_for_move()

    def wait_for_move(self) -> None:
        self._run_async_gather("run_speed_to_position", lambda _: None)

    @property
    def is_moving(self) -> bool:
        """Returns True if any motor is moving."""
        return any(motor.is_moving for motors in self.axes.values() for motor in motors)

    @property
    def position(self) -> list:
        return [motor.position for motors in self.axes.values() for motor in motors]

    def _run_async_gather(self, fn: str, callback: Callable[[list], Any]):
        """Runs the specified function on all motors asynchronously."""
        # TODO: can re remove DummyStepperMotor dependence?
        fns = [
            getattr(motor, fn)
            for motors in self.axes.values()
            for motor in motors
            if not isinstance(motor, DummyStepperMotor)
        ]
        return call_async_gather(fns, callback)

    def __getattr__(self, name: str) -> Any:
        """This is a passthrough to the underlying motor objects."""
        results, fns = [], []
        for motors in self.axes.values():
            motor_results, motor_fns = [], []
            for motor in motors:
                # Will throw attribute error if the attribute is not found
                attr = getattr(motor, name)

                # If the attr is a method, we'll accumulate the fns and call them with
                # gather
                if callable(attr):
                    motor_fns.append(attr)
                else:
                    motor_results.append(attr)
            if motor_results:
                results.append(motor_results)
            if motor_fns:
                fns.append(motor_fns)

        if fns:

            def wrapper(*args):
                items = []
                if len(args) == len(fns):
                    items = zip(fns, [[arg] for arg in args])
                elif len(args) <= 1:
                    items = [(fn, args) for fn in fns]
                else:
                    raise ValueError(f"Invalid number of arguments: {args}, {fns}")

                results = []
                for fn, args in items:
                    if isinstance(fn, list):
                        results.append([motor_fn(*args) for motor_fn in fn])
                    else:
                        results.append(fn(*args))
                return results

            return wrapper
        else:
            return results

    @property
    def axes(self) -> dict[StepperMotorSystemAxis, list[StepperMotor]]:
        """Returns the axes of the stepper motor system."""
        return self._axes

    def close(self):
        get_logger().info("Closing steppers...")
        if "_axes" not in self.__dict__:
            get_logger().warning("StepperMotorSystem has not been initialized.")
            return

        for motors in self.axes.values():
            for motor in motors:
                motor.close()


class DummyStepperSystem(StepperMotorSystem):
    """A dummy stepper system that does nothing."""

    def __init__(self):
        super().__init__(StepperMotorSystemConfig.create(axes={}))

    def move_to(self, *positions: float):
        pass

    def move_by(self, *positions: float):
        pass

    def wait_for_move(self):
        pass

    def close(self):
        pass
