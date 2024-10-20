from pylablib.devices.Thorlabs import KinesisMotor

from cc_hardware.utils.logger import get_logger
from cc_hardware.drivers.stepper_motor import StepperMotor


class KinesisStepperMotor(StepperMotor):
    def __init__(self, port: str, is_rack_system: bool = True, scale: float = 1.0):
        self._is_okay = False
        self._scale = scale

        self._motor = KinesisMotor(port, is_rack_system=is_rack_system)
        self._motor.open()
        self._is_okay = True
        get_logger().info(f"Connected to Kinesis motor on {port}")

    def close(self, home: bool = True):
        if not self.is_okay:
            return

        try:
            if home:
                self.home()
            self._motor.close()
            get_logger().info("Kinesis motor disconnected.")
        except Exception as e:
            get_logger().error(f"Failed to disconnect the Kinesis motor: {e}")

    def home(self):
        if not self.is_okay:
            return

        try:
            self._motor.home()
            get_logger().info("Kinesis motor homed.")
        except Exception as e:
            get_logger().error(f"Failed to home the Kinesis motor: {e}")
            self.close(home=False)

    def move_by(self, relative_position: float):
        if not self.is_okay:
            get_logger().warning("Kinesis motor is not operational.")
            return

        try:
            self._motor.move_by(self._convert(relative_position))
            self._motor.wait_for_stop()
            get_logger().info(f"Rotated by {relative_position} degrees.")
        except Exception as e:
            get_logger().error(f"Failed to rotate the Kinesis motor: {e}")
            self.close()

    def move_to(self, position: float):
        if not self.is_okay:
            get_logger().warning("Kinesis motor is not operational.")
            return

        try:
            self._motor.move_to(self._convert(position))
            self._motor.wait_for_stop()
            get_logger().info(f"Rotated to {position} degrees.")
        except Exception as e:
            get_logger().error(f"Failed to rotate the Kinesis motor: {e}")
            self.close()

    @property
    def is_okay(self):
        return self._is_okay

    def _convert(self, position: float):
        return position * self._scale
