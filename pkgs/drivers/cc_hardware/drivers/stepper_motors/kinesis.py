from pylablib.devices.Thorlabs import KinesisMotor

from cc_hardware.utils.logger import get_logger
from cc_hardware.drivers.stepper_motor import StepperMotor


class KinesisStepperMotor(StepperMotor):
    """This class is an abstraction above a Kinesis motor object.

    Args:
        port (str): The port of the Kinesis motor.

    Keyword Args:
        is_rack_system (bool): Whether the motor is a rack system. Defaults to True.
        scale (float): The scale factor for the motor. Defaults to 1.0.
        channel (int): The channel of the motor. Defaults to 1.
        lower_limit (float): The lower limit of the motor. Defaults to None.
        upper_limit (float): The upper limit of the motor. Defaults to None.
        clip_at_limits (bool): When a position is passed that exceeds the limits, if
            True, the motor will move to the limit. If False, the motor will print an
            error and not move. Defaults to False.
    """

    def __init__(
        self,
        port: str,
        *,
        channel: int = 1,
        is_rack_system: bool = True,
        scale: float = 1.0,
    ):
        self._is_okay = False
        self._scale = scale
        self._lower_limit = None
        self._upper_limit = None
        self._clip_at_limits = False

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
        *,
        max_velocity: float | None = None,
        acceleration: float | None = None,
        lower_limit: float | None = None,
        upper_limit: float | None = None,
        clip_at_limits: bool = False,
        initial_position: float | None = None,
        home: bool = False,
        check_homed: bool | None = None,
        reference_position: float | None = None,
    ) -> bool:
        if not self.is_okay:
            get_logger().warning("Kinesis motor is not operational.")
            return False

        self._lower_limit = lower_limit
        self._upper_limit = upper_limit
        self._clip_at_limits = (lower_limit or upper_limit) and clip_at_limits

        check_homed = check_homed if check_homed is not None else not home

        try:
            # Set velocity and acceleration
            self._motor.setup_velocity(
                acceleration=acceleration, max_velocity=max_velocity
            )

            # Homing sequence
            if home:
                self.home()
            if check_homed and not self._motor.is_homed():
                get_logger().error("Kinesis motor is not homed.")
                return False

            # Set reference and initial position
            if reference_position is not None:
                self._motor.set_position_reference(reference_position)
            if initial_position is not None:
                self.move_to(initial_position)

            get_logger().info("Kinesis motor initialized.")
        except Exception as e:
            get_logger().error(f"Failed to initialize the Kinesis motor: {e}")
            self.close(home=False)

        return self.is_okay

    def close(self, home: bool = False):
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

    def home(self, **kwargs):
        if not self.is_okay:
            return

        try:
            self._motor.home(**kwargs)
            get_logger().info("Kinesis motor homed.")
        except Exception as e:
            get_logger().error(f"Failed to home the Kinesis motor: {e}")
            self.close(home=False)

    def move_by(self, relative_position: float, wait_for_stop: bool = True):
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
        """Checks if the position is within the limits."""
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
        return self._is_okay

    @property
    def position(self) -> float:
        if not self.is_okay:
            return 0.0

        return self._convert_from(self._motor.get_position())

    @property
    def lower_limit(self) -> float | None:
        return self._lower_limit

    @lower_limit.setter
    def lower_limit(self, value: float | None):
        self._lower_limit = value

    @property
    def upper_limit(self) -> float | None:
        return self._upper_limit

    @upper_limit.setter
    def upper_limit(self, value: float | None):
        self._upper_limit = value

    def _convert_to(self, position: float) -> float:
        return position * self._scale

    def _convert_from(self, position: float) -> float:
        return position / self._scale


class KinesisRotationStage(KinesisStepperMotor):
    IS_RACK_SYSTEM = True
    SCALE = 75000

    # Parameters taken from SCurve profile
    ACCELERATION = 1877344.2032468664
    MAX_VELOCITY = 3755159.538002981

    def __init__(
        self,
        *args,
        is_rack_system: bool | None = None,
        scale: int | None = None,
        **kwargs,
    ):
        if is_rack_system is None:
            is_rack_system = self.IS_RACK_SYSTEM
        kwargs.setdefault("is_rack_system", is_rack_system)
        if scale is None:
            scale = self.SCALE
        kwargs.setdefault("scale", scale)

        super().__init__(*args, **kwargs)

    def initialize(self, **kwargs) -> bool:
        kwargs.setdefault("max_velocity", self.MAX_VELOCITY)
        kwargs.setdefault("acceleration", self.ACCELERATION)

        return super().initialize(**kwargs)
