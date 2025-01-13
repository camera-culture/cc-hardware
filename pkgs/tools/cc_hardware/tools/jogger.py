"""Joystick-like interface for controlling the gantry system."""

import curses
import logging
import sys
from collections import deque

from cc_hardware.drivers.stepper_motors import StepperMotorSystem
from cc_hardware.tools.app import APP, typer
from cc_hardware.utils.logger import get_logger

# ======================

jogger_APP = typer.Typer()
APP.add_typer(jogger_APP, name="jogger")

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
    def __init__(self, system: str, port: str):
        self._system = StepperMotorSystem.create_from_registry(system, port)
        self._system.initialize()
        assert len(self._system.position) == 2, "Only 2D systems are supported."

        self._scale = 1.0

        # Set up output buffer
        self.output_buffer = deque(maxlen=1000)  # Increased maxlen for more lines

        # Set up logging
        self.log_handler = LogBufferHandler(self.output_buffer)

    def start(self):
        curses.wrapper(self._start)

    def _start(self, stdscr: curses.window):
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

        # Calculate output window height to fill remaining space
        output_height = max_y - main_ui_height
        if output_height < 5:
            output_height = 5  # Set a minimum height for the output window
            main_ui_height = max_y - output_height

        # Create the output window
        self.output_win = curses.newwin(output_height, max_x, main_ui_height, 0)

        try:
            while self._step(stdscr, main_ui_lines, main_ui_height):
                pass
        finally:
            # Restore stdout and stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

            self._system.close()

    def _step(self, stdscr: curses.window, main_ui_lines, main_ui_height) -> bool:
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
            self._scale = min(int(self._scale), 1)

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


@jogger_APP.command()
def run(
    system: StepperMotorSystem.registered,
    port: str | None = None,
    exit_immediately: bool = False,
):
    jogger = Jogger(system.value, port)
    if exit_immediately:
        return
    jogger.start()


# ======================


def jogger():
    import argparse

    parser = argparse.ArgumentParser(
        description="Communication script between a PC and the Arduino. "
        "For controlling linear slides with a CNCShield."
    )

    parser.add_argument(
        "--port",
        type=str,
        help="The port the arduino is on. If None, auto port detection is used.",
        default="/dev/ttyUSB0",
    )
    parser.add_argument(
        "-E" "--exit-immediately",
        dest="exit_immediately",
        action="store_true",
        help="Exit immediately after instantiating the gantry object.",
    )

    args = parser.parse_args()

    jogger = Jogger(args.port)
    if args.exit_immediately:
        return
    jogger.start()


if __name__ == "__main__":
    jogger()
