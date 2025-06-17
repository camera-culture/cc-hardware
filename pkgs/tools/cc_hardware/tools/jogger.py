"""Joystick-like interface for controlling the gantry system."""

import curses
import logging
import sys
from collections import deque

from cc_hardware.drivers.stepper_motors import (
    StepperMotorSystem,
    StepperMotorSystemConfig,
)
from cc_hardware.utils import get_logger, register_cli, run_cli

# ======================


class OutputCapture:
    """Captures stdout and stderr output and stores it in a buffer."""

    def __init__(self, buffer):
        self.buffer: list[str] = buffer

    def write(self, s: str):
        for line in s.rstrip().split("\n"):
            self.buffer.append(line.rstrip())

    def flush(self):
        pass


class LogBufferHandler(logging.Handler):
    """Logging handler that stores log records in a buffer."""

    def __init__(self, buffer):
        super().__init__()
        self.buffer: list[str] = buffer

    def emit(self, record):
        msg = self.format(record)
        self.buffer.append(msg)


# ======================


class Jogger:
    def __init__(self, stepper_system: StepperMotorSystem):
        self._system = stepper_system
        self._system.initialize()
        assert len(self._system.position) == 2, "Only 2D systems are supported."

        self._scale = 1.0

        # Set up output buffer
        self.output_buffer = deque(maxlen=1000)  # Increased maxlen for more lines

        # Set up logging
        self.log_handler = LogBufferHandler(self.output_buffer)

        self._stdscr: curses.window = None  # Will be set by curses.wrapper
        self._main_ui_lines: str = None

    def run(self):
        def _run(stdscr: curses.window):
            self.start(stdscr)
            try:
                while self._step(self._stdscr, self._main_ui_lines):
                    pass
            finally:
                # Restore stdout and stderr
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__

                self._system.close()

        curses.wrapper(_run)

    def start(self, stdscr: curses.window | None = None):
        if stdscr is None:
            if self._stdscr is None:
                self._stdscr = curses.initscr()
                curses.noecho()
                curses.cbreak()
                self._stdscr.keypad(True)

            stdscr = self._stdscr
        self._stdscr = stdscr

        curses.curs_set(0)  # Disable blinking cursor
        stdscr.nodelay(True)  # Make getch non-blocking

        # Redirect stdout and stderr
        sys.stdout = OutputCapture(self.output_buffer)
        sys.stderr = OutputCapture(self.output_buffer)

        # Get screen size
        max_y, max_x = stdscr.getmaxyx()

        # Determine the height of the main UI
        main_ui_lines = [
            "Robot Teleop GUI",
            f"Target Position: X={self.x:.2f}, Y={self.y:.2f}",
            f"Scale: {self._scale:.2f}",
            "",
            "Use arrow keys to move",
            "Press 'H' for Home (reset position)",
            "Press 'I' to increase scale, 'D' to decrease scale",
            "Press 'Q' to quit",
        ]
        main_ui_height = len(main_ui_lines) + 2  # Add padding if necessary

        self._main_ui_lines = main_ui_lines

        # Calculate output window height to fill remaining space
        output_height = max_y - main_ui_height
        if output_height < 5:
            output_height = 5  # Set a minimum height for the output window
            main_ui_height = max_y - output_height

        # Create the output window
        self.output_win = curses.newwin(output_height, max_x, main_ui_height, 0)

    def update(self) -> bool:
        assert self._stdscr is not None, "Jogger must be started with start() first."
        try:
            return self._step(self._stdscr, self._main_ui_lines)
        finally:
            # Ensure the output window is refreshed after each step
            self.output_win.refresh()

            # Restore the cursor position
            self._stdscr.move(0, 0)
            self._stdscr.refresh()

    def _step(self, stdscr: curses.window, main_ui_lines: int) -> bool:
        key = stdscr.getch()
        curses.flushinp()

        if key == ord("q") or key == ord("Q"):
            return False
        elif key == curses.KEY_UP:
            self.y += self._scale
        elif key == curses.KEY_DOWN:
            self.y -= self._scale
        elif key == curses.KEY_LEFT:
            self.x -= self._scale
        elif key == curses.KEY_RIGHT:
            self.x += self._scale
        elif key == ord("h") or key == ord("H"):
            self.home()
        elif key == ord("i") or key == ord("I"):
            self._scale *= 2
        elif key == ord("d") or key == ord("D"):
            self._scale /= 2
            self._scale = max(int(self._scale), 1)

        # Update the UI
        stdscr.erase()
        # Update dynamic content in main_ui_lines
        main_ui_lines[1] = f"Target Position: X={self.x:.2f}, Y={self.y:.2f}"
        main_ui_lines[2] = f"Scale: {self._scale:.2f}"

        for idx, line in enumerate(main_ui_lines):
            stdscr.addstr(idx, 0, line)

        stdscr.refresh()

        # Update the output window
        self.output_win.erase()
        self.output_win.border()  # Draw a border around the output window

        # Add title to the output window
        self.output_win.addstr(0, 2, " Output ")  # Position the title on the top border

        # Calculate the maximum number of lines and columns inside the border
        max_output_lines, max_output_cols = self.output_win.getmaxyx()
        max_output_lines -= 2  # Adjust for border
        max_output_cols -= 2  # Adjust for border

        # Combine log messages and output buffer
        combined_buffer = list(self.output_buffer)
        # Get the last max_output_lines lines
        display_lines = combined_buffer[-max_output_lines:]
        for idx, line in enumerate(display_lines):
            # Truncate line to fit in the window
            if len(line) > max_output_cols:
                line = line[:max_output_cols]
            # Add text inside the border
            self.output_win.addstr(idx + 1, 1, line)
        self.output_win.refresh()

        # Limit screen update rate
        curses.napms(10)

        return self._system.is_okay

    def home(self):
        self._system.home()
        self.set_position(0, 0)

    @property
    def x(self):
        return self._system.position[0]

    @x.setter
    def x(self, value):
        delta = value - self.x
        self.set_position(delta, 0)

    @property
    def y(self):
        return self._system.position[1]

    @y.setter
    def y(self, value):
        delta = value - self.y
        self.set_position(0, delta)

    @property
    def xy(self):
        return self.x, self.y

    @xy.setter
    def xy(self, xy):
        x, y = xy
        dx, dy = x - self.x, y - self.y
        self.set_position(dx, dy)

    def set_position(self, dx, dy):
        get_logger().info(f"Moving by {dx}, {dy}...")
        self._system.move_by(dx, dy)


# ======================


@register_cli
def jogger(
    stepper_system: StepperMotorSystemConfig,
):
    from cc_hardware.utils.manager import Manager

    def setup(manager: Manager):
        _stepper_system = StepperMotorSystem.create_from_config(stepper_system)
        manager.add(stepper_system=_stepper_system)

        jogger = Jogger(_stepper_system)
        jogger.start()
        manager.add(jogger=jogger)

    def loop(
        iter: int, manager: Manager, stepper_system: StepperMotorSystem, jogger: Jogger
    ) -> bool:
        return jogger.update()

    def cleanup(manager: Manager, stepper_system: StepperMotorSystem, jogger: Jogger):
        get_logger().info("Cleaning up...")
        stepper_system.move_to(0, 0)
        stepper_system.close()

    with Manager() as manager:
        manager.run(setup=setup, loop=loop, cleanup=cleanup)


if __name__ == "__main__":
    run_cli(jogger)
