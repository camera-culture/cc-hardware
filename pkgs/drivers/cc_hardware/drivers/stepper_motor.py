from abc import ABC, abstractmethod

class StepperMotor(ABC):
    """This class is an abstraction above a stepper motor object."""

    @abstractmethod
    def close(self) -> None:
        """Closes the stepper motor."""
        pass

    @abstractmethod
    def home(self) -> None:
        """Homes the stepper motor."""
        pass

    @abstractmethod
    def move_to(self, position: float) -> None:
        """Moves the stepper motor to a position."""
        pass

    @abstractmethod
    def move_by(self, relative_position: float) -> None:
        """Moves the stepper motor by a relative position."""
        pass

    @property
    @abstractmethod
    def is_okay(self) -> bool:
        """Checks if the stepper motor is operational."""
        pass