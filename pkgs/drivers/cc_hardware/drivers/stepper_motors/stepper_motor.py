from abc import ABC, abstractmethod

class StepperMotor(ABC):
    """
    An abstract base class for controlling a stepper motor. This class provides a 
    unified interface for common operations such as moving to a specific position, 
    homing, and closing the motor. It also includes a property to check the operational 
    status of the motor.

    Any subclass must implement all the defined abstract methods to ensure 
    compatibility with the expected motor control behavior.
    """

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
