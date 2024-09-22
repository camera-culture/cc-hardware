import csv
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

from cc_hardware.cnc_robot.gantry import Gantry
from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.transformations import Action, GlobalFrame, LocalFrame


class MotionController(ABC):
    """This is the base controller class. Inherit from this and implement the step
    method to create a new controller.

    Arguments:
        gantry (Gantry): The gantry to actually step.
        logfile (Path | None): The file to log states to. The logfile will be a csv with
            (x, y, t) at each location by default.
        num_steps (int): The number of steps to perform for the entire experiment.
        global_frame (GlobalFrame): The frame transform to the global frame, i.e.
            the ground truth position of the controller.
        init_action (Action): The initial action in the local frame.
    """

    def __init__(
        self,
        gantry: Gantry,
        logfile: Path | None,
        num_steps: int,
        global_frame: GlobalFrame,
        init_action: LocalFrame,
    ):
        self._gantry = gantry
        self._num_axes = self._gantry.num_axes
        self._logfile = logfile
        self._num_steps = num_steps

        self._global_frame = global_frame
        self._init_action = init_action
        self._current_frame: GlobalFrame = global_frame.copy()

        # We want to start with a fresh csv, so remove the logfile if it exists.
        if self._logfile is not None:
            self._logfile.parent.mkdir(parents=True, exist_ok=True)
            if self._logfile.exists():
                self._logfile.unlink()
            self._logfile.touch()

    def initialize(self):
        """Called before running the capture loop. Can be used to home the gantry."""
        # Move to the initial position and save it
        get_logger().info(f"Moving to initial position {self._init_action}...")
        self.move(self._init_action)
        self.save(self._current_frame)

    def step(self, index: int) -> bool:
        """Calculates the next step of the gantry given the current index and the
        current position. The current position and the timestamp is also saved to a
        csv.

        Arguments:
            index (int): The current step number. Index should be between 0 and
                `num_steps`.

        Returns:
            bool: Returns True if the controller has more steps to perform, False
                otherwise.
        """
        get_logger().debug(f"Stepping {index}...")

        if index >= self._num_steps:
            get_logger().info("Finished stepping.")
            return False

        # Get the current frame and the next step
        action = self.get_next_action(index)

        # Move the gantry
        new_local_frame = self._current_frame @ self._global_frame.inverse() @ action
        get_logger().info(f"Moving to {new_local_frame}...")
        self.move(action)

        # Save the position; retrieve the new position in case it's not the same
        self.save(self._current_frame)
        get_logger().info(f"Done moving to {new_local_frame}.")

        get_logger().debug(f"Finished stepping {index}.")

        return True

    @abstractmethod
    def get_next_action(self, index: int) -> Action:
        """Get the next step for the gantry. This is called by step to move the
        gantry.

        Arguments:
            index (int): The current step number. Index should be between 0 and
                `num_steps`.

        Returns:
            Action: The next action to take.
        """
        pass

    def move(self, action: Action):
        self._gantry.set_position(*action.get())

        # Update the current frame
        self._current_frame = self._current_frame.apply(action, T=self._global_frame)

    def save(self, frame: GlobalFrame):
        if self._logfile is None:
            return

        t = time.time()

        get_logger().debug(f"Saving state to {self._logfile}...")
        with open(self._logfile, "a") as f:
            writer = csv.writer(f)
            formatted_data = [f"{x:.6f}" for x in [t, *frame.mat.flatten()]]
            writer.writerow(formatted_data)

    def shutdown(self):
        """Turns off the controller. Basically just tears down the gantry."""
        # Home the gantry
        home_pos = (self._current_frame @ self._global_frame.inverse()).inverse()
        get_logger().info(f"Homing gantry to {home_pos}...")
        self.move(Action.from_frame(home_pos))

    def close(self):
        """Closes the controller. This is called after the capture loop has finished."""
        self.shutdown()


class RectangleController(MotionController):
    """This is square controller, as in it moves along the boundary of a rectangle
    with size as defined by `x_range` and `y_range`.

    Keyword Arguments:
        x_range (Tuple[float, float]): The range in x for the rectangle.
        y_range (Tuple[float, float]): The range in y for the rectangle.
        reverse (bool): If True, will move in the opposite direction.
    """

    def __init__(
        self,
        *,
        gantry: Gantry,
        logfile: Path | None = None,
        num_steps: int,
        global_frame: GlobalFrame,
        init_action: LocalFrame,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        x_steps_per_direction: int,
        y_steps_per_direction: int,
        reverse: bool = False,
    ):
        super().__init__(gantry, logfile, num_steps, global_frame, init_action)

        assert (
            x_steps_per_direction * y_steps_per_direction == num_steps
        ), "x_steps_per_direction * y_steps_per_direction must equal num_steps"
        self._x_steps_per_direction = x_steps_per_direction
        self._y_steps_per_direction = y_steps_per_direction

        self._x_range, self._y_range = x_range, y_range
        self._x_step = (x_range[1] - x_range[0]) / x_steps_per_direction
        self._y_step = (y_range[1] - y_range[0]) / y_steps_per_direction
        self._reverse = reverse

    def get_next_action(self, index: int) -> Action:
        if self._reverse:
            if index < self._y_steps_per_direction:
                # Move up
                return Action.create(y=self._y_step)
            elif index < self._y_steps_per_direction + self._x_steps_per_direction:
                # Move right
                return Action.create(x=self._x_step)
            elif index < self._y_steps_per_direction * 2 + self._x_steps_per_direction:
                # Move down
                return Action.create(y=-self._y_step)
            elif (
                index
                < self._y_steps_per_direction * 2 + self._x_steps_per_direction * 2
            ):
                # Move left
                return Action.create(x=-self._x_step)
            else:
                raise Exception("This should never happen")
        else:
            if index < self._x_steps_per_direction:
                # Move right
                return Action.create(x=self._x_step)
            elif index < self._x_steps_per_direction + self._y_steps_per_direction:
                # Move up
                return Action.create(y=self._y_step)
            elif index < self._x_steps_per_direction * 2 + self._y_steps_per_direction:
                # Move left
                return Action.create(x=-self._x_step)
            elif (
                index
                < self._x_steps_per_direction * 2 + self._y_steps_per_direction * 2
            ):
                # Move down
                return Action.create(y=-self._y_step)
            else:
                raise Exception("This should never happen")


class LinearController(MotionController):
    """This controller will simply move along an axis in either one or both directions.

    Keyword Arguments:
        axis_range (Tuple[float, float]): The range of the axis to move. The gantry
            will be homed at axis_range[0].
        action_key (str): The key in the position command to move. For example, if
            action_key is 'y', then the gantry will move along y.
        steps_per_direction (Optional[int]): The number of steps in each direction to
            move (i.e. back and forth). num_steps must be divisible by steps per
            direction. If set to None, will move num_steps just in one direction.
    """

    def __init__(
        self,
        *,
        gantry: Gantry,
        logfile: Path | None = None,
        num_steps: int,
        global_frame: GlobalFrame,
        init_action: LocalFrame,
        axis_range: Tuple[float, float],
        action_key: str,
        steps_per_direction: Optional[int] = None,
    ):
        steps_per_direction = steps_per_direction or num_steps
        assert (
            num_steps % steps_per_direction == 0
        ), "num_steps must be divisible by steps_per_direction"

        super().__init__(gantry, logfile, num_steps, global_frame, init_action)

        self._axis_range = axis_range
        self._steps_per_direction = steps_per_direction
        self._action_key = action_key

        self._step = (axis_range[1] - axis_range[0]) / steps_per_direction

        # Used to keep track of direction. Will flip at the end of a range to go back
        # the other way (if desired). Initialized to -1 since we'll flip it on the
        # first call to get_next_action since index % steps_per_direction is 0.
        self._direction = -1

    def get_next_action(self, index: int) -> Action:
        if index % self._steps_per_direction == 0:
            self._direction *= -1

        # Returns 0 for the ignored axes, step otherwise
        return Action.create(**{self._action_key: self._direction * self._step})


class SnakeController(MotionController):
    """
    This controller zigzags along the x and y axes. It moves in the x direction for
    `x_steps_per_direction` steps, then moves in the y direction once, then negative x
    for `x_steps_per_direction` steps, then moves in the y direction once, etc.
    NOTE: `x_steps_per_direction` * `y_steps_per_direction` must equal `num_steps`.

    Keyword Arguments:
        x_range (Tuple[float, float]): The range of the x axis to move.
        y_range (Tuple[float, float]): The range of the y axis to move.
        x_steps_per_direction (int): The number of steps to move in the x direction
            before moving in the y direction.
        y_steps_per_direction (int): The number of steps to move in the y direction
            before moving in the x direction.
    """

    def __init__(
        self,
        *,
        gantry,
        logfile: Path | None = None,
        global_frame: GlobalFrame = GlobalFrame.create(),
        init_action: LocalFrame = LocalFrame.create(),
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        x_steps_per_direction: int,
        y_steps_per_direction: int,
    ):
        num_steps = x_steps_per_direction * y_steps_per_direction
        super().__init__(gantry, logfile, num_steps, global_frame, init_action)

        assert (
            x_steps_per_direction * y_steps_per_direction == num_steps
        ), "x_steps_per_direction * y_steps_per_direction must equal num_steps"
        self._x_steps_per_direction = x_steps_per_direction
        self._y_steps_per_direction = y_steps_per_direction

        self._x_range, self._y_range = x_range, y_range
        self._x_step = (x_range[1] - x_range[0]) / (x_steps_per_direction - 1)
        self._y_step = (y_range[1] - y_range[0]) / (y_steps_per_direction - 1)

    def get_next_action(self, index: int) -> Action:
        if index == self._num_steps - 1:
            return Action.create()

        x_step, y_step = 0, 0
        if index // self._x_steps_per_direction % 2 == 0:
            if index % self._x_steps_per_direction == self._x_steps_per_direction - 1:
                y_step = self._y_step
            else:
                x_step = self._x_step

        else:
            if index % self._x_steps_per_direction == self._x_steps_per_direction - 1:
                y_step = self._y_step
            else:
                x_step = -self._x_step

        return Action.create(x=x_step, y=y_step)


class StaticController(MotionController):
    """This represents a static motion profile (as in no movement). Supports x,y or
    x,y,z."""

    def __init__(
        self,
        *,
        gantry: Gantry,
        logfile: Path | None = None,
        num_steps: int,
        global_frame: GlobalFrame,
        init_action: LocalFrame,
        **kwargs,
    ):
        super().__init__(gantry, logfile, num_steps, global_frame, init_action)

    def get_next_action(self, *_) -> Action:
        return Action.create()
