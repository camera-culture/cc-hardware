"""This module contains the TelemetrixStepperMotor and TelemetrixStepperMotorSystem
classes which are wrappers around the Telemetrix library's interface with stepper
motors. These classes provide a unified interface for controlling stepper motors
connected to a CNCShield using the Telemetrix library."""

import inspect
from functools import partial
from typing import Any

from telemetrix import telemetrix

from cc_hardware.drivers.stepper_motors import (
    StepperMotor,
    StepperMotorSystem,
    StepperMotorSystemAxis,
)
from cc_hardware.utils import call_async, get_logger, register

# ======================


@register
class TelemetrixStepperMotor(StepperMotor):
    """This is a wrapper of the Telemetrix library's interface with stepper motors.

    NOTE: Initialization of this class effectively homes the motor. Call
    `set_current_position` to explicitly set the current position.

    Args:
        board (telemetrix.Telemetrix): The Telemetrix board object

    Keyword Args:
        distance_pin (int): The pin on the CNCShield that controls this motor's position
        direction_pin (int): The pin on the CNCShield that controls this motor's
            position
        enable_pin (int): The pin that controls this motor's enable pin..
        cm_per_rev (float): The number of centimeters per revolution of the motor.
        steps_per_rev (int): The number of steps per revolution of the motor.
        speed (float): The speed of the motor in cm/s.
        flip_direction (bool): If True, the motor will move in the opposite direction.
    """

    def __init__(
        self,
        board: telemetrix.Telemetrix,
        *,
        distance_pin: int,
        direction_pin: int,
        enable_pin: int,
        cm_per_rev: float,
        steps_per_rev: int,
        speed: float,
        flip_direction: bool = False,
    ):
        self._board = board
        self._cm_per_rev = cm_per_rev
        self._steps_per_rev = steps_per_rev
        self._speed = speed
        self._flip_direction = flip_direction

        # Create the motor instance and set initial settings
        self._motor = board.set_pin_mode_stepper(pin1=distance_pin, pin2=direction_pin)
        get_logger().info(f"Created TelemetrixStepperMotor with id {self._motor}")
        self.set_enable_pin(enable_pin)
        self.set_3_pins_inverted(enable=True)

        # Set constants and home the motor
        self.set_max_speed(self._speed)
        self.set_current_position(0)

        # Initialize the motor
        self.set_target_position_cm(0)
        call_async(self.run_speed_to_position, lambda _: None)

        get_logger().debug(f"Initialized TelemetrixStepperMotor with id {self.id}")

    def home(self) -> None:
        """Homes the stepper motor to its reference or zero position."""
        self.move_to(0)

    def move_to(self, position: float) -> None:
        """Moves the stepper motor to a specific absolute position.

        Args:
            position (float): The target absolute position to move the motor to.
        """
        self.set_absolute_target_position_cm(position)

    def move_by(self, relative_position: float) -> None:
        """Moves the stepper motor by a specified relative amount from its current
        position.

        Args:
            relative_position (float): The amount to move the motor by, relative to its
                current position.
        """
        self.set_target_position_cm(relative_position)

    def wait_for_move(self) -> None:
        """Waits for the motor to complete its current move operation."""
        call_async(self.run_speed_to_position, lambda *_: None)

    @property
    def position(self) -> float:
        """Returns the current position of the stepper motor."""

        def get_position(data):
            f = -1 if self._flip_direction else 1
            return self.revs_to_cm(data[2]) * f

        return call_async(self.get_current_position, get_position)

    @property
    def id(self) -> int:
        """Returns the motor's id."""
        return self._motor

    @property
    def flip_direction(self) -> bool:
        return self._flip_direction

    def set_target_position_cm(self, relative_cm: float):
        """Sets the target position of the motor relative to current position.

        Args:
            relative_cm (float): The relative position to move the motor to.
        """
        get_logger().info(f"Setting target position to {relative_cm} cm...")
        relative_steps = self.cm_to_revs(relative_cm)
        if self._flip_direction:
            relative_steps *= -1
        self.move(relative_steps)
        self.set_speed(self._speed)  # Need to set speed again since move overwrites it

    def set_absolute_target_position_cm(self, position_cm: float):
        """Sets the absolute target position in cm."""
        get_logger().info(f"Setting absolute target position to {position_cm} cm...")
        steps = self.cm_to_revs(position_cm)
        if self._flip_direction:
            steps *= -1
        self._board.stepper_move_to(self._motor, steps)
        self.set_speed(self._speed)  # Need to set speed again since move overwrites it

    def cm_to_revs(self, cm: float) -> int:
        """Converts cm to steps."""
        return int(cm / self._cm_per_rev * self._steps_per_rev)

    def revs_to_cm(self, revs: int) -> float:
        """Converts steps to cm."""
        return revs / self._steps_per_rev * self._cm_per_rev

    @property
    def is_moving(self) -> bool:
        """Checks if the stepper motor is currently in motion."""
        def is_moving(data):
            return data[2] == 1
        return call_async(self.is_running, is_moving)

    @property
    def is_okay(self) -> bool:
        """Checks if the stepper motor is in a healthy operational state."""
        return self._board is not None

    def close(self) -> None:
        """Closes the connection or shuts down the stepper motor safely."""
        if "_board" in self.__dict__:
            self.stop()

    def __getattr__(self, key: str) -> Any:
        """This is a passthrough to the underlying stepper object.

        Usually, stepper methods are accessed through the board with stepper_*. You
        can access these methods directly here using motorX.target_position(...) which
        equates to motorX._board.stepper_target_position(...). Also, if these methods
        require a motor as input, we'll pass it in.
        """
        # Will throw an AttributeError if the attribute doesn't exist in board
        attr = getattr(self._board, f"stepper_{key}", None) or getattr(self._board, key)

        # If "motor_id" is in the signature of the method, we'll pass the motor id to
        # the method. This will return False if the attr isn't a method.
        signature = inspect.signature(attr)
        if signature.parameters.get("motor_id", None):
            return partial(attr, self._motor)
        else:
            return attr


# ======================


class TelemetrixStepperMotorSystem(StepperMotorSystem):
    """This is a wrapper of the Telemetrix library's interface with multiple
    stepper motors.

    Args:
        port (str | None): The port to connect to the Telemetrix board. If None,
            the port will be attempted to be auto-detected.

    Keyword Args:
        axes (dict[str, list[TelemetrixStepperMotor]]): A dictionary of axes and
            the motors that are attached to them. The key is the axis name and the
            value is a list of motors attached to that axis.
        kwargs: Additional keyword arguments to pass to the Telemetrix board.
    """

    def __init__(
        self,
        port: str | None = None,
        *,
        axes: dict[str, list[TelemetrixStepperMotor]],
        **kwargs,
    ):
        # This is the arduino object. Initialize it once. If port is None, the library
        # will attempt to auto-detect the port.
        self._board = telemetrix.Telemetrix(port, **kwargs)

        # Update the axes to include the board
        axes = {
            axis: [motor(self._board) for motor in motors]
            for axis, motors in axes.items()
        }

        # Initialize the multi-axis stepper system
        super().__init__(axes)

    @property
    def is_okay(self) -> bool:
        return True

    def home(self):
        raise NotImplementedError(f"Cannot home {self.__class__.__name__}.")

    def close(self) -> None:
        """Closes the connection or shuts down the stepper motor safely."""
        if hasattr(self, "_board"):
            super().close()
            self._board.shutdown()
            del self._board


# ======================

# These are stepper motors with pins assigned on a CNCShield
# https://www.makerstore.com.au/wp-content/uploads/filebase/publications/CNC-Shield-Guide-v1.0.pdf?srsltid=AfmBOooLloq8m90Yut7SvV4jDVqMSgQVeiFeI7rdsZYqy9eaUdIjHfCf


@register
class TelemetrixStepperMotorX(TelemetrixStepperMotor):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("distance_pin", 2)
        kwargs.setdefault("direction_pin", 5)
        kwargs.setdefault("enable_pin", 8)
        super().__init__(*args, **kwargs)


@register
class TelemetrixStepperMotorY(TelemetrixStepperMotor):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("distance_pin", 3)
        kwargs.setdefault("direction_pin", 6)
        kwargs.setdefault("enable_pin", 8)
        super().__init__(*args, **kwargs)


@register
class TelemetrixStepperMotorZ(TelemetrixStepperMotor):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("distance_pin", 4)
        kwargs.setdefault("direction_pin", 7)
        kwargs.setdefault("enable_pin", 8)
        super().__init__(*args, **kwargs)


@register
class TelemetrixStepperMotorXReversed(TelemetrixStepperMotorX):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("flip_direction", True)
        super().__init__(*args, **kwargs)


@register
class TelemetrixStepperMotorYReversed(TelemetrixStepperMotorY):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("flip_direction", True)
        super().__init__(*args, **kwargs)


@register
class TelemetrixStepperMotorZReversed(TelemetrixStepperMotorZ):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("flip_direction", True)
        super().__init__(*args, **kwargs)


# ======================


@register
class DualDrive2AxisGantry_X(TelemetrixStepperMotorZ):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cm_per_rev", 2.8)
        kwargs.setdefault("steps_per_rev", 200)
        kwargs.setdefault("speed", 1000)
        super().__init__(*args, **kwargs)


@register
class DualDrive2AxisGantry_Y1(TelemetrixStepperMotorY):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cm_per_rev", 4)
        kwargs.setdefault("steps_per_rev", 200)
        kwargs.setdefault("speed", 500)
        super().__init__(*args, **kwargs)


@register
class DualDrive2AxisGantry_Y2(TelemetrixStepperMotorXReversed):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cm_per_rev", 4)
        kwargs.setdefault("steps_per_rev", 200)
        kwargs.setdefault("speed", 500)
        super().__init__(*args, **kwargs)


@register
class DualDrive2AxisGantry(TelemetrixStepperMotorSystem):
    def __init__(self, *args, **kwargs):
        axes = {
            StepperMotorSystemAxis.X: [DualDrive2AxisGantry_X],
            StepperMotorSystemAxis.Y: [
                DualDrive2AxisGantry_Y1,
                DualDrive2AxisGantry_Y2,
            ],
        }
        super().__init__(*args, axes=axes, **kwargs)


# ======================


@register
class SingleDrive1AxisGantry_X(TelemetrixStepperMotorX):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cm_per_rev", 2.8)
        kwargs.setdefault("steps_per_rev", 2850)
        kwargs.setdefault("speed", 2**15 - 1)
        super().__init__(*args, **kwargs)


@register
class SingleDrive1AxisGantry_Y(TelemetrixStepperMotorY):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cm_per_rev", 2.8)
        kwargs.setdefault("steps_per_rev", 2850)
        kwargs.setdefault("speed", 2**15 - 1)
        super().__init__(*args, **kwargs)


@register
class SingleDrive1AxisGantry(TelemetrixStepperMotorSystem):
    def __init__(self, *args, **kwargs):
        axes = {
            StepperMotorSystemAxis.X: [SingleDrive1AxisGantry_X],
            StepperMotorSystemAxis.Y: [SingleDrive1AxisGantry_Y],
        }
        super().__init__(*args, axes=axes, **kwargs)
