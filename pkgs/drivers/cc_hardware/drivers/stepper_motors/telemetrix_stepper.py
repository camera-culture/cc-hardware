"""This module contains the TelemetrixStepperMotor and TelemetrixStepperMotorSystem
classes which are wrappers around the Telemetrix library's interface with stepper
motors. These classes provide a unified interface for controlling stepper motors
connected to a CNCShield using the Telemetrix library."""

import inspect
from functools import partial
from typing import Any
from dataclasses import field

from telemetrix import telemetrix

from cc_hardware.drivers.stepper_motors import (
    StepperMotor,
    StepperMotorConfig,
    StepperMotorSystemConfig,
    StepperMotorSystem,
    StepperMotorSystemAxis,
)
from cc_hardware.utils import call_async, get_logger, config_wrapper, II

# ======================


@config_wrapper
class TelemetrixStepperMotorConfig(StepperMotorConfig):
    """Configuration for Telemetrix stepper motors.

    Attributes:
        board (telemetrix.Telemetrix | None): The Telemetrix board instance. Optional
            since we set the board in the system.

        distance_pin (int): The pin on the CNCShield that controls this motor's position
        direction_pin (int): The pin on the CNCShield that controls this motor's
            position
        enable_pin (int): The pin that controls this motor's enable pin.
        cm_per_rev (float): The number of centimeters per revolution of the motor.
        steps_per_rev (int): The number of steps per revolution of the motor.
        speed (int): The speed of the motor in cm/s.
        flip_direction (bool): If True, the motor will move in the opposite direction.
    """

    board: telemetrix.Telemetrix | None = None

    distance_pin: int
    direction_pin: int
    enable_pin: int
    cm_per_rev: float
    steps_per_rev: int
    speed: int
    flip_direction: bool = False


class TelemetrixStepperMotor[T: TelemetrixStepperMotorConfig](StepperMotor[T]):
    """This is a wrapper of the Telemetrix library's interface with stepper motors.

    NOTE: Initialization of this class effectively homes the motor. Call
    `set_current_position` to explicitly set the current position.
    """

    def __init__(
        self,
        config: T,
    ):
        super().__init__(config)

        # Create the motor instance and set initial settings
        self._motor = config.board.set_pin_mode_stepper(
            pin1=config.distance_pin, pin2=config.direction_pin
        )
        get_logger().info(f"Created TelemetrixStepperMotor with id {self._motor}")
        self.set_enable_pin(config.enable_pin)
        self.set_3_pins_inverted(enable=True)

        # Set constants and home the motor
        self.set_max_speed(self.config.speed)
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
            f = -1 if self.config.flip_direction else 1
            return self.revs_to_cm(data[2]) * f

        return call_async(self.get_current_position, get_position)

    @property
    def id(self) -> int:
        """Returns the motor's id."""
        return self._motor

    @property
    def flip_direction(self) -> bool:
        return self.config.flip_direction

    def set_target_position_cm(self, relative_cm: float):
        """Sets the target position of the motor relative to current position.

        Args:
            relative_cm (float): The relative position to move the motor to.
        """
        get_logger().info(f"Setting target position to {relative_cm} cm...")
        relative_steps = self.cm_to_revs(relative_cm)
        if self.flip_direction:
            relative_steps *= -1
        self.move(relative_steps)
        self.set_speed(
            self.config.speed
        )  # Need to set speed again since move overwrites it

    def set_absolute_target_position_cm(self, position_cm: float):
        """Sets the absolute target position in cm."""
        get_logger().info(f"Setting absolute target position to {position_cm} cm...")
        steps = self.cm_to_revs(position_cm)
        if self.flip_direction:
            steps *= -1
        self.config.board.stepper_move_to(self._motor, steps)
        self.set_speed(
            self.config.speed
        )  # Need to set speed again since move overwrites it

    def cm_to_revs(self, cm: float) -> int:
        """Converts cm to steps."""
        return int(cm / self.config.cm_per_rev * self.config.steps_per_rev)

    def revs_to_cm(self, revs: int) -> float:
        """Converts steps to cm."""
        return revs / self.config.steps_per_rev * self.config.cm_per_rev

    @property
    def is_moving(self) -> bool:
        """Checks if the stepper motor is currently in motion."""

        def is_moving(data):
            return data[2] == 1

        return call_async(self.is_running, is_moving)

    @property
    def is_okay(self) -> bool:
        """Checks if the stepper motor is in a healthy operational state."""
        return self.config.board is not None

    def close(self) -> None:
        """Closes the connection or shuts down the stepper motor safely."""
        if hasattr(self, "_motor"):
            self.stop()
            del self._motor

    def __getattr__(self, key: str) -> Any:
        """This is a passthrough to the underlying stepper object.

        Usually, stepper methods are accessed through the board with stepper_*. You
        can access these methods directly here using motorX.target_position(...) which
        equates to motorX._board.stepper_target_position(...). Also, if these methods
        require a motor as input, we'll pass it in.
        """
        # Will throw an AttributeError if the attribute doesn't exist in board
        attr = getattr(self.config.board, f"stepper_{key}", None) or getattr(
            self.config.board, key
        )

        # If "motor_id" is in the signature of the method, we'll pass the motor id to
        # the method. This will return False if the attr isn't a method.
        signature = inspect.signature(attr)
        if signature.parameters.get("motor_id", None):
            return partial(attr, self._motor)
        else:
            return attr


# ======================


@config_wrapper
class TelemetrixStepperMotorSystemConfig(StepperMotorSystemConfig):
    """Configuration for Telemetrix stepper motor systems.

    Attributes:
        port (str | None): The port to connect to the Telemetrix board. If None,
            the port will be attempted to be auto-detected.
        arduino_wait (int): The time to wait for the Arduino to initialize in seconds.
    """

    port: str | None = None
    arduino_wait: int = 4


class TelemetrixStepperMotorSystem[T: TelemetrixStepperMotorSystemConfig](
    StepperMotorSystem[T]
):
    """This is a wrapper of the Telemetrix library's interface with multiple
    stepper motors.
    """

    def __init__(
        self,
        config: T,
    ):
        # This is the arduino object. Initialize it once. If port is None, the library
        # will attempt to auto-detect the port.
        self._board = telemetrix.Telemetrix(
            config.port, arduino_wait=config.arduino_wait
        )

        for motors in config.axes.values():
            for motor in motors:
                motor.board = self._board

        # Initialize the multi-axis stepper system
        super().__init__(config)

    @property
    def is_okay(self) -> bool:
        return all(motor.is_okay for motors in self.axes.values() for motor in motors)

    def home(self):
        raise NotImplementedError(f"Cannot home {self.__class__.__name__}.")

    def close(self) -> None:
        """Closes the connection or shuts down the stepper motor safely."""
        super().close()
        if hasattr(self, "_board"):
            self._board.shutdown()
            del self._board


# ======================

# These are stepper motors with pins assigned on a CNCShield
# https://www.makerstore.com.au/wp-content/uploads/filebase/publications/CNC-Shield-Guide-v1.0.pdf?srsltid=AfmBOooLloq8m90Yut7SvV4jDVqMSgQVeiFeI7rdsZYqy9eaUdIjHfCf


@config_wrapper
class TelemetrixStepperMotorXConfig(TelemetrixStepperMotorConfig):
    distance_pin: int = 2
    direction_pin: int = 5
    enable_pin: int = 8


@config_wrapper
class TelemetrixStepperMotorYConfig(TelemetrixStepperMotorConfig):
    distance_pin: int = 3
    direction_pin: int = 6
    enable_pin: int = 8


@config_wrapper
class TelemetrixStepperMotorZConfig(TelemetrixStepperMotorConfig):
    distance_pin: int = 4
    direction_pin: int = 7
    enable_pin: int = 8


# ======================


@config_wrapper
class DualDrive2AxisGantryXConfig(TelemetrixStepperMotorZConfig):
    cm_per_rev: float = 2.8
    steps_per_rev: int = 200
    speed: int = 1000


@config_wrapper
class DualDrive2AxisGantryY1Config(TelemetrixStepperMotorYConfig):
    cm_per_rev: float = 4
    steps_per_rev: int = 200
    speed: int = 500


@config_wrapper
class DualDrive2AxisGantryY2Config(TelemetrixStepperMotorXConfig):
    cm_per_rev: float = 4
    steps_per_rev: int = 200
    speed: int = 500

    # Flipped direction for Y2 to match Y1
    flip_direction: bool = True


@config_wrapper
class DualDrive2AxisGantryConfig(TelemetrixStepperMotorSystemConfig):
    axes: dict[StepperMotorSystemAxis, list[TelemetrixStepperMotorConfig]] = {
        StepperMotorSystemAxis.X: [DualDrive2AxisGantryXConfig],
        StepperMotorSystemAxis.Y: [
            DualDrive2AxisGantryY1Config,
            DualDrive2AxisGantryY2Config,
        ],
    }


# ======================


@config_wrapper
class SingleDrive1AxisGantryXConfig(TelemetrixStepperMotorXConfig):
    cm_per_rev: float = 2.8
    steps_per_rev: int = 2850
    speed: int = 2**15 - 1


@config_wrapper
class SingleDrive1AxisGantryYConfig(TelemetrixStepperMotorYConfig):
    cm_per_rev: float = 2.8
    steps_per_rev: int = 2850
    speed: int = 2**15 - 1


@config_wrapper
class SingleDrive1AxisGantryConfig(TelemetrixStepperMotorSystemConfig):
    axes: dict[StepperMotorSystemAxis, list[TelemetrixStepperMotorConfig]] = field(
        default_factory=lambda: {
            StepperMotorSystemAxis.X: [SingleDrive1AxisGantryXConfig],
            StepperMotorSystemAxis.Y: [SingleDrive1AxisGantryYConfig],
        }
    )
