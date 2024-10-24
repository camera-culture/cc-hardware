from pylablib.devices.Thorlabs import KinesisMotor

from cc_hardware.utils.logger import get_logger
from cc_hardware.drivers.stepper_motors.stepper_motor import StepperMotor


class KinesisStepperMotor(StepperMotor):
    """
    A wrapper class for controlling a Kinesis motor using the StepperMotor interface.
    This class provides additional features such as homing, movement with limits, and
    scaling of motor positions.

    Args:
        port (str): The port of the Kinesis motor.

    Keyword Args:
        is_rack_system (bool): Whether the motor is part of a rack system. Defaults to
            True.
        scale (float): The scale factor for motor movement. Defaults to 1.0.
        channel (str): The channel of the motor. Defaults to 1.
        lower_limit (float, optional): The lower movement limit of the motor. Defaults
            to None.
        upper_limit (float, optional): The upper movement limit of the motor. Defaults
            to None.
        clip_at_limits (bool): If True, the motor will clip movements at the set limits.
            If False, an error will be logged if the limits are exceeded. Defaults to
            False.
    """

    def __init__(
        self,
        port: str,
        *,
        is_rack_system: bool = True,
        scale: float = 1.0,
        channel: str = 1,
        lower_limit: float | None = None,
        upper_limit: float | None = None,
        clip_at_limits: bool = False,
    ):
        self._is_okay = False
        self._scale = scale
        self._lower_limit = lower_limit
        self._upper_limit = upper_limit
        self._clip_at_limits = (lower_limit or upper_limit) and clip_at_limits

        self._motor = KinesisMotor(
            port, is_rack_system=is_rack_system, default_channel=channel
        )
        self._motor.open()
        if not self._motor._is_channel_enabled():
            get_logger().error("Failed to connect to Kinesis motor.")
            self.close()
            return
        self._is_okay = True
        get_logger().info(f"Connected to Kinesis motor on {port}")

    def initialize(
        self,
        inital_position: float | None = None,
        *,
        home: bool = False,
        check_homed: bool | None = None,
        reference_position: float | None = None,
    ) -> bool:
        """
        Initialize the Kinesis motor, with options to home and set a reference position.

        Args:
            inital_position (float, optional): The initial position to move the motor
                to. Defaults to None.
            home (bool): Whether to home the motor during initialization. Defaults to
                False.
            check_homed (bool, optional): Whether to check if the motor is homed.
                Defaults to the opposite of `home`.
            reference_position (float, optional): The reference position to set.
                Defaults to None.

        Returns:
            bool: True if the motor is successfully initialized, False otherwise.
        """
        if not self.is_okay:
            get_logger().warning("Kinesis motor is not operational.")
            return False

        check_homed = check_homed if check_homed is not None else not home

        try:
            if home:
                self.home()
            if check_homed and not self._motor.is_homed():
                get_logger().error("Kinesis motor is not homed.")
                return False
            if reference_position is not None:
                self._motor.set_position_reference(reference_position)
            if inital_position is not None:
                self.move_to(inital_position)
            get_logger().info("Kinesis motor initialized.")
        except Exception as e:
            get_logger().error(f"Failed to initialize the Kinesis motor: {e}")
            self.close(home=False)

        return self.is_okay

    def close(self, home: bool = False):
        """
        Closes the Kinesis motor connection, with an optional homing operation.

        Args:
            home (bool): Whether to home the motor before closing. Defaults to False.
        """
        if not self.is_okay:
            return

        self._motor.stop()

        try:
            if home:
                self.home()
            self._motor.close()
            get_logger().info("Kinesis motor disconnected.")
        except Exception as e:
            get_logger().error(f"Failed to disconnect the Kinesis motor: {e}")

        self._is_okay = False

    def home(self):
        """
        Homes the Kinesis motor to its reference or zero position.
        """
        if not self.is_okay:
            return

        try:
            self._motor.home()
            get_logger().info("Kinesis motor homed.")
        except Exception as e:
            get_logger().error(f"Failed to home the Kinesis motor: {e}")
            self.close(home=False)

    def move_by(self, relative_position: float, wait_for_stop: bool = True):
        """
        Moves the motor by a specified relative position.

        Args:
            relative_position (float): The amount to move the motor by, relative to
                its current position.
            wait_for_stop (bool): Whether to wait for the motor to stop after moving.
                Defaults to True.
        """
        if not self.is_okay:
            get_logger().warning("Kinesis motor is not operational.")
            return

        try:
            relative_position = self._check_limits(relative_position, self.position)
            if relative_position is None:
                return

            self._motor.move_by(self._convert_to(relative_position))
            if wait_for_stop:
                self._motor.wait_for_stop()
            get_logger().info(f"Rotated by {relative_position} degrees.")
        except Exception as e:
            get_logger().error(f"Failed to rotate the Kinesis motor: {e}")
            self.close()

    def move_to(self, position: float, wait_for_stop: bool = True):
        """
        Moves the motor to a specified absolute position.

        Args:
            position (float): The target absolute position to move the motor to.
            wait_for_stop (bool): Whether to wait for the motor to stop after moving.
                Defaults to True.
        """
        if not self.is_okay:
            get_logger().warning("Kinesis motor is not operational.")
            return

        try:
            position = self._check_limits(position)
            if position is None:
                return

            self._motor.move_to(self._convert_to(position))
            if wait_for_stop:
                self._motor.wait_for_stop()
            get_logger().info(f"Rotated to {position} degrees.")
        except Exception as e:
            get_logger().error(f"Failed to rotate the Kinesis motor: {e}")
            self.close()

    def _check_limits(
        self, position: float, current_position: float = 0
    ) -> float | None:
        """
        Checks if a given position is within the defined limits and clips the position
        if necessary.

        Args:
            position (float): The target position to check.
            current_position (float): The current position of the motor. Defaults to 0.

        Returns:
            float | None: The validated position within limits, or None if the
                position exceeds limits and clipping is disabled.
        """
        if (
            self._lower_limit is not None
            and position + current_position < self._lower_limit
        ):
            if self._clip_at_limits:
                return self._lower_limit
            get_logger().error("Position is below the lower limit.")
            return None
        elif (
            self._upper_limit is not None
            and position + current_position > self._upper_limit
        ):
            if self._clip_at_limits:
                return self._upper_limit
            get_logger().error("Position is above the upper limit.")
            return None
        return position

    @property
    def is_okay(self) -> bool:
        """
        Check if the motor is in a healthy operational state.

        Returns:
            bool: True if the motor is operational, False otherwise.
        """
        return self._is_okay

    @property
    def position(self) -> float:
        """
        Get the current absolute position of the motor.

        Returns:
            float: The current position of the motor, or 0.0 if the motor is not
                operational.
        """
        if not self.is_okay:
            return 0.0

        return self._convert_from(self._motor.get_position())

    @property
    def lower_limit(self) -> float | None:
        """
        Get or set the lower movement limit of the motor.

        Returns:
            float | None: The current lower limit, or None if not set.
        """
        return self._lower_limit

    @lower_limit.setter
    def lower_limit(self, value: float | None):
        self._lower_limit = value

    @property
    def upper_limit(self) -> float | None:
        """
        Get or set the upper movement limit of the motor.

        Returns:
            float | None: The current upper limit, or None if not set.
        """
        return self._upper_limit

    @upper_limit.setter
    def upper_limit(self, value: float | None):
        self._upper_limit = value

    def _convert_to(self, position: float) -> float:
        """
        Convert a given position to the internal motor scale.

        Args:
            position (float): The position to convert.

        Returns:
            float: The converted position.
        """
        return position * self._scale

    def _convert_from(self, position: float) -> float:
        """
        Convert a given internal motor position to the user-defined scale.

        Args:
            position (float): The position to convert.

        Returns:
            float: The converted position.
        """
        return position / self._scale
