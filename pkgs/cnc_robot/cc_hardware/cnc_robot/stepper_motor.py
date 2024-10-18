import inspect
from functools import partial
from typing import Any, Callable, List, TypeAlias

from telemetrix import telemetrix

from cc_hardware.cnc_robot.utils import call_async
from cc_hardware.utils.logger import get_logger


class StepperMotor:
    """This is a wrapper of the telemetrix library's interface with stepper motors.

    NOTE: Initialization of this class effectively homes the motor. Call
    `set_current_position` to explicitly set the current position.

    Args:
        board (telemetrix.Telemetrix): The telemetrix board object
        distance_pin (int): The pin on the CNCShield that controls this motor's position
        direction_pin (int): The pin on the CNCShield that controls this motor's
            position

    Keyword Args:
        enable_pin (int): The pin on the CNCShield that controls this motor's enable
            pin. Defaults to 8.
        flip_direction (bool): If True, the motor will move in the opposite direction.
    """

    def __init__(
        self,
        board: telemetrix.Telemetrix,
        distance_pin: int,
        direction_pin: int,
        *,
        cm_per_rev: float,
        steps_per_rev: int,
        speed: float,
        enable_pin: int = 8,
        flip_direction: bool = False,
    ):
        self._board = board
        self._cm_per_rev = cm_per_rev
        self._steps_per_rev = steps_per_rev
        self._speed = speed
        self._flip_direction = flip_direction

        # Create the motor instance and sets some settings
        self._motor = board.set_pin_mode_stepper(pin1=distance_pin, pin2=direction_pin)
        self.set_enable_pin(enable_pin)
        self.set_3_pins_inverted(enable=True)

        # Set constants and home the motor
        self.set_max_speed(self._speed)
        self.set_current_position(0)

        # We have to initialize the motor for some reason to have it work
        self.set_target_position_cm(0)
        call_async(self.run_speed_to_position, lambda _: None)

    @property
    def id(self) -> int:
        """Returns the motor's id."""
        return self._motor

    @property
    def flip_direction(self) -> bool:
        return self._flip_direction

    def set_target_position_cm(self, relative_cm: int):
        """This set's the target position of the motor. This is the position the motor
        will move to when `run_speed_to_position` is called.

        Args:
            relative_cm (int): The relative position to move the motor to.
        """
        relative_revs = self.cm_to_revs(relative_cm)
        if self._flip_direction:
            relative_revs *= -1
        self.move(relative_revs)
        self.set_speed(self._speed)  # need to set speed again since move overwrites it

    def cm_to_revs(self, cm: float) -> int:
        """Converts cm to revolutions."""
        return int(cm / self._cm_per_rev * self._steps_per_rev)

    def revs_to_cm(self, revs: int) -> float:
        """Converts revolutions to cm."""
        return revs / self._steps_per_rev * self._cm_per_rev

    def __getattr__(self, key: str) -> Any:
        """This is a passthrough to the underlying stepper object.

        Usually, stepper methods are accessed through the board with stepper_*. You
        can access these methods directly here using motorX.target_position(...) which
        equates to motorX._board.stepper_target_position(...). Also, if these methods
        require a motor as input, we'll pass it in
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

    def __del__(self):
        """Disables the motor."""
        if "_board" in self.__dict__:
            self.stop()


class DummyStepperMotor:
    """This is a dummy stepper motor class that does nothing. This is useful for testing
    or when you don't have a CNCShield connected to the computer. Also can be used for
    axes which don't have a motor attached to them."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name: str) -> Any:
        def noop(*args, **kwargs) -> Any:
            pass

        return noop

    def __del__(self):
        pass


StepperMotorPartial: TypeAlias = Callable[[telemetrix.Telemetrix], "StepperMotor"]


class GroupedStepperMotor:
    def __init__(
        self, board: telemetrix.Telemetrix, motors: List[StepperMotorPartial], **kwargs
    ):
        self._motors = [motor(board, **kwargs) for motor in motors]

    def __getattr__(self, name: str) -> Any:
        """This is a passthrough to the underlying motor objects."""
        results, fns = [], []
        for motor in self._motors:
            # Will throw attribute error if the attribute is not found
            attr = getattr(motor, name)

            # If the attr is a method, we'll accumulate the fns and call them with
            # gather
            if callable(attr):
                fns.append(attr)
            else:
                results.append(attr)

        if fns:

            def wrapper(*args):
                items = []
                if len(args) == len(fns):
                    items = zip(fns, [[arg] for arg in args])
                elif len(args) == 1:
                    items = [(fn, args) for fn in fns]
                else:
                    raise ValueError("Invalid number of arguments")

                results = []
                for fn, args in items:
                    results.append(fn(*args))
                return results

            return wrapper
        else:
            return results


StepperMotorX = partial(StepperMotor, distance_pin=2, direction_pin=5)
StepperMotorY = partial(StepperMotor, distance_pin=3, direction_pin=6)
StepperMotorZ = partial(StepperMotor, distance_pin=4, direction_pin=7)


class StepperMotorFactory:
    X = StepperMotorX
    Y = StepperMotorY
    Z = StepperMotorZ

    NEG_X = partial(StepperMotorX, flip_direction=True)
    NEG_Y = partial(StepperMotorY, flip_direction=True)
    NEG_Z = partial(StepperMotorZ, flip_direction=True)

    DUMMY = DummyStepperMotor

    @staticmethod
    def grouped(*motors: StepperMotorPartial, **kwargs) -> StepperMotorPartial:
        return partial(GroupedStepperMotor, motors=motors, **kwargs)


__call__ = [
    StepperMotor,
    StepperMotorFactory,
    StepperMotorPartial,
    GroupedStepperMotor,
]
