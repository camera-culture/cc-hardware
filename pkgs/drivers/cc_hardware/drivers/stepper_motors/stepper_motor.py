from abc import ABC, abstractmethod
from typing import Any

from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.registry import Registry, register


class StepperMotor(Registry, ABC):
    """
    An abstract base class for controlling a stepper motor. This class provides a
    unified interface for common operations such as moving to a specific position,
    homing, and closing the motor. It also includes a property to check the operational
    status of the motor.

    Any subclass must implement all the defined abstract methods to ensure
    compatibility with the expected motor control behavior.
    """

    def initialize(self):
        get_logger().info(f"Initialized {self.__class__.__name__}.")

    @abstractmethod
    def close(self) -> None:
        """
        Closes the connection or shuts down the stepper motor safely. Implementations
        should ensure that the motor is properly powered down and any resources are
        released to avoid damage or memory leaks.
        """
        pass

    @abstractmethod
    def home(self) -> None:
        """
        Homes the stepper motor to its reference or zero position. This method should
        move the motor to a predefined starting point, which could involve moving
        until a limit switch or sensor is triggered to establish a known starting
        position.
        """
        pass

    @abstractmethod
    def move_to(self, position: float) -> None:
        """
        Moves the stepper motor to a specific absolute position.

        Args:
            position (float): The target absolute position to move the motor to. The
                interpretation of this value may depend on the specific implementation
                and motor characteristics (e.g., steps, angle).
        """
        pass

    @abstractmethod
    def move_by(self, relative_position: float) -> None:
        """
        Moves the stepper motor by a specified relative amount from its current
        position.

        Args:
            relative_position (float): The amount to move the motor by, relative to its
                current position. This could represent steps, degrees, or any other
                unit, depending on the motor's configuration.
        """
        pass

    @abstractmethod
    def wait_for_move(self) -> None:
        """
        Waits for the motor to complete its current move operation. This method should
        block the execution until the motor has reached its target position or
        completed the current motion command.
        """
        pass

    @property
    @abstractmethod
    def position(self) -> float:
        """
        Returns the current position of the stepper motor. The position value should
        represent the motor's current location in the same units as the move_to and
        move_by methods.

        Returns:
            float: The current position of the motor.
        """
        pass

    @property
    @abstractmethod
    def is_okay(self) -> bool:
        """
        Checks if the stepper motor is in a healthy operational state. This could
        involve verifying that the motor is not in an error state, is receiving power,
        and has no detected hardware issues.

        Returns:
            bool: True if the motor is operational, False otherwise.
        """
        pass


@register
class DummyStepperMotor:
    """This is a dummy stepper motor class that does nothing. This is useful for testing
    or when you don't have the stepper connected to the computer. Also can be used for
    axes which don't have a motor attached to them."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name: str) -> Any:
        def noop(*args, **kwargs) -> Any:
            pass

        return noop

    def __del__(self):
        pass
