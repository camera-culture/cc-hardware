from enum import Enum
from typing import Any, overload

from cc_hardware.drivers.stepper_motors import DummyStepperMotor, StepperMotor
from cc_hardware.utils.asyncio_utils import call_async_gather
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.registry import Registry, register

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


# ======================


@register
class StepperMotorSystem(Registry, StepperMotor):
    """This is a wrapper around multiple stepper motors which defines the system
    as a whole (i.e. a gantry or multi-axis rotation stage).

    Args:
        axes (dict[StepperMotorSystemAxis, list[StepperMotor]]): A dictionary of axes
            and the motors that are attached to them.
    """

    def __init__(
        self,
        axes: dict[StepperMotorSystemAxis, list[StepperMotor]],
    ):
        # The current state of the motor is considered zero (as in the motor is homed).
        self._axes = axes

    @overload
    def move_to(self, *positions: float):
        assert len(positions) == len(
            self._axes
        ), f"Got {len(positions)} positions, expected {len(self._axes)}"

        self.move_to({axis: pos for axis, pos in zip(self._axes, positions)})

    @overload
    def move_to(self, **positions: float):
        """Move to the specified position.

        Grab the current position and subtract the target position to get the relative
        position. Then move the motors by the relative position.
        """
        assert len(positions) == len(
            self._axes
        ), f"Got {len(positions)} positions, expected {len(self._axes)}"
        current_positions = self.position
        positions = {
            axis: pos - current_pos
            for (axis, pos), current_pos in zip(positions.items(), current_positions)
        }

        self.move_by(**positions)

    @overload
    def move_by(self, *positions: float):
        ...

    @overload
    def move_by(self, **positions: float):
        ...

    def move_by(self, *args: float, **kwargs: float):
        """Moves the steppers to the specified positions."""
        if args and kwargs:
            raise ValueError("move_to takes either all positional or all keyword args.")
        elif args:
            assert len(args) == len(
                self._axes
            ), f"Got {len(args)} args, expected {len(self._axes)}"
            positions = {axis: position for axis, position in zip(self._axes, args)}
        elif kwargs:
            assert len(kwargs) == len(
                self._axes
            ), f"Got {len(kwargs)} kwargs, expected {len(self._axes)}"
            positions = {
                StepperMotorSystemAxis[axis.upper()]: pos
                for axis, pos in kwargs.items()
            }

        # Set the target position of each motor
        for axis, position in positions.items():
            # This doesn't actually move the motor to the position, just set's the
            # target position. The motor won't move until run_speed_to_position or
            # run_speed is called.
            for motor in self._axes[axis]:
                motor.move_to(position)

    def wait_for_move(self) -> None:
        # Wait for all motors to complete their motion
        self._run_async_gather("wait_for_move", lambda _: None)

    @property
    def position(self) -> list:
        return [[motor.position for motor in motors] for motors in self._axes.values()]

    def _run_async_gather(self, fn: str, callback: callable[[list], Any]):
        """Runs the specified function on all motors asynchronously."""
        # TODO: can re remove DummyStepperMotor dependence?
        fns = [
            getattr(motor, fn)
            for motors in self._axes.values()
            for motor in motors
            if not isinstance(motor, DummyStepperMotor)
        ]
        return call_async_gather(fns, callback)

    def __getattr__(self, name: str) -> Any:
        """This is a passthrough to the underlying motor objects."""
        results, fns = [], []
        for motors in self._axes.values():
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

    def close(self):
        get_logger().info("Closing steppers...")
        if "_axes" in self.__dict__:
            for motors in self._axes.values():
                for motor in motors:
                    motor.close()
