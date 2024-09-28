from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, overload

import numpy as np
from telemetrix import telemetrix

from cc_hardware.cnc_robot.stepper_motor import (
    DummyStepperMotor,
    StepperMotor,
    StepperMotorFactory,
    StepperMotorPartial,
)
from cc_hardware.cnc_robot.utils import call_async_gather
from cc_hardware.utils.logger import get_logger


class Axis(Enum):
    X = "X"
    Y = "Y"
    Z = "Z"
    ROLL = "ROLL"
    PITCH = "PITCH"
    YAW = "YAW"


class Gantry:
    """This is a wrapper around multiple stepper motors which defines the system
    as a whole (i.e. a gantry).

    Args:
        port (str | None): The port the arduino is on. If None, auto port detection
            is used.

    Keyword Args:
        axes (List[AxisPartial]): The axes which make up this gantry. An axis may
            include multiple motors.
    """

    def __init__(
        self,
        port: str | None = None,
        *,
        axes: Dict[Axis, List[StepperMotorPartial]],
        home: Tuple[float] | None = None,
    ):
        # This is the arduino object. Initialize it once. If port is None, the library
        # will attempt to auto-detect the port.
        self._board = telemetrix.Telemetrix(port)

        # Create the motors objects.
        # Must initialize the motors after telemetrix
        # The current state of the motor is considered zero (as in the motor is homed).
        self._axes = {
            axis: [motor(self._board) for motor in motors]
            for axis, motors in axes.items()
        }

        # If home is set, we'll home the gantry here. The gantry is assumed to start at
        # the origin, and this will reposition the gantry to home  and set the internal
        # position to (0, 0).
        if home is not None:
            self.set_position(*home)
            self.set_current_position(*np.zeros_like(home))

    @overload
    def set_position(self, *positions_cm: float):
        ...

    @overload
    def set_position(self, **positions_cm: float):
        ...

    def set_position(self, *args: float, **kwargs: float):
        """Set the position of the arm. This method will move the arm to the specified
        position. All axes are moved simultaneously. This method blocks until the motion
        is completed.

        All units in cm!
        """
        if args and kwargs:
            raise ValueError(
                "set_position takes either all positional or all keyword args."
            )
        elif args:
            assert len(args) == len(
                self._axes
            ), f"Got {len(args)} args, expected {len(self._axes)}"
            positions = {axis: position for axis, position in zip(self._axes, args)}
        elif kwargs:
            assert len(kwargs) == len(
                self._axes
            ), f"Got {len(kwargs)} kwargs, expected {len(self._axes)}"
            positions = {Axis[axis.upper()]: pos for axis, pos in kwargs.items()}

        # Set the target position of each motor
        for axis, position_cm in positions.items():
            # This doesn't actually move the motor to the position, just set's the
            # target position. The motor won't move until run_speed_to_position or
            # run_speed is called.
            for motor in self._axes[axis]:
                motor.set_target_position_cm(position_cm)

        # Wait for all motors to complete their motion
        self._run_async_gather("run_speed_to_position", lambda _: None)

    def get_position(self, *, check_positions: bool = False) -> np.ndarray:
        current_positions = self._run_async_gather(
            "get_current_position", lambda d: self.motor(d[1]).revs_to_cm(d[2])
        )
        i = 0
        positions = []
        for motors in self._axes.values():
            axis_positions = []
            for motor in motors:
                f = -1 if motor._flip_direction else 1
                axis_positions.append(current_positions[i] * f)
                i += 1
            assert not check_positions or all(
                abs(p) == abs(axis_positions[0]) for p in axis_positions
            ), axis_positions
            positions.append(axis_positions[0])
        return np.array(positions)

    def motor(self, motor_id: int) -> StepperMotor:
        for motors in self._axes.values():
            for motor in motors:
                if motor.id == motor_id:
                    return motor
        else:
            raise ValueError(f"Motor with id {motor_id} not found")

    def _run_async_gather(self, fn: str, callback: Callable[[List], Any]):
        """Runs the specified function on all motors asynchronously."""
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

    @property
    def num_axes(self) -> int:
        return len(self._axes)

    @property
    def axes(self) -> List[str]:
        return list([axis.name for axis in self._axes])

    def close(self):
        get_logger().info("Closing gantry...")
        if "_axes" in self.__dict__:
            # Must come before board shutdown
            del self._axes
        if "_board" in self.__dict__ and self._board is not None:
            self._board.shutdown()
            del self._board

    def __del__(self):
        self.close()


class DummyGantry:
    """This is a dummy gantry object that can be used for testing purposes. It does not
    actually initialize a telemetrix object."""

    def __init__(
        self,
        init_pos: List[float] = (0, 0, 0, 0, 0, 0),
        *,
        override_getattr: bool = True,
        **kwargs,
    ):
        self.set_current_position(*init_pos)
        self.override_getattr = override_getattr

    def set_current_position(self, *pos: float):
        self.pos = np.array(pos, dtype=float)

    def set_position(self, *step: float):
        step = np.array(list(step) + [0] * (6 - len(step)))
        self.pos += step

    def get_position(self) -> np.ndarray:
        return self.pos

    @property
    def num_axes(self) -> int:
        return len(self.pos)

    @property
    def axes(self) -> List[Axis]:
        return list(Axis)

    def close(self):
        pass

    def __getattr__(self, name: str) -> Any:
        """This is a passthrough to the underlying motor objects."""

        if self.override_getattr:

            def noop(*args, **kwargs):
                pass

            return noop


NEMA_17_SMALL_PULLEY = dict(cm_per_rev=2.8, steps_per_rev=200, speed=1000)
NEMA_17_LARGE_PULLEY = dict(cm_per_rev=4, steps_per_rev=200, speed=500)

DualDrive2AxisGantry: Type[Gantry] = partial(
    Gantry,
    axes={
        Axis.X: [partial(StepperMotorFactory.NEG_Z, **NEMA_17_SMALL_PULLEY)],
        Axis.Y: [
            partial(StepperMotorFactory.NEG_X, **NEMA_17_LARGE_PULLEY),
            partial(StepperMotorFactory.Y, **NEMA_17_LARGE_PULLEY),
        ],
        Axis.Z: [partial(StepperMotorFactory.DUMMY)],
        Axis.ROLL: [partial(StepperMotorFactory.DUMMY)],
        Axis.PITCH: [partial(StepperMotorFactory.DUMMY)],
        Axis.YAW: [partial(StepperMotorFactory.DUMMY)],
    },
)
"""Two steppers actuate the same axis and are plugged into 'X' and 'Y', respectively.
One stepper actuates the other axis and is plugged into 'Z'."""

NEMA_17_SMALL = dict(cm_per_rev=2.8, steps_per_rev=2850, speed=2**15 - 1)

SingleDrive1AxisGantry: Type[Gantry] = partial(
    Gantry,
    axes={
        Axis.X: [partial(StepperMotorFactory.X, **NEMA_17_SMALL)],
        Axis.Y: [partial(StepperMotorFactory.Y, **NEMA_17_SMALL)],
        Axis.Z: [partial(StepperMotorFactory.DUMMY)],
        Axis.ROLL: [partial(StepperMotorFactory.DUMMY)],
        Axis.PITCH: [partial(StepperMotorFactory.DUMMY)],
        Axis.YAW: [partial(StepperMotorFactory.DUMMY)],
    },
)
"""Each stepper actuates its own axis. One stepper is plugged into 'X' and the other is
plugged into 'Y'."""


class GantryFactory:
    GANTRY_MAP: Dict[str, Type[Gantry]] = {
        "DualDrive2AxisGantry": DualDrive2AxisGantry,
        "SingleDrive1AxisGantry": SingleDrive1AxisGantry,
        "DummyGantry": DummyGantry,
    }

    @staticmethod
    def create(name: str, *args, **kwargs) -> Gantry:
        if name not in GantryFactory.GANTRY_MAP:
            raise ValueError(
                f"Invalid gantry name: {name}. "
                f"Must be one of {list(GantryFactory.GANTRY_MAP.keys())}."
            )
        return GantryFactory.GANTRY_MAP[name](*args, **kwargs)


__all__ = [
    "GantryFactory",
]
