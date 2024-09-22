"""This script is used to "jog" the gantry to different positions. You can use
this to position the gantry at a specific location or move it in general."""

import curses

from cc_hardware.cnc_robot.gantry import GantryFactory


class Jogger:
    def __init__(self, gantry: str, port: str | None):
        self._gantry = GantryFactory.create(gantry, port)

        self._target_x, self._target_y = 0, 0
        self._scale = 1

    def start(self):
        curses.wrapper(self._start)

    def _start(self, stdscr: curses.window):
        curses.curs_set(0)  # disable blinking cursor
        stdscr.nodelay(True)  # make getch non-blocking

        while self._step(stdscr):
            pass

    def _step(self, stdscr: curses.window) -> bool:
        key = stdscr.getch()
        curses.flushinp()

        if key == ord("q") or key == ord("Q"):
            self.xy = (0, 0)
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
            self._scale = int(self._scale)

        # Update the UI
        stdscr.erase()
        stdscr.addstr(0, 0, "Robot Teleop GUI")
        stdscr.addstr(2, 0, f"Target Position: X={self.x}, Y={self.y}")
        stdscr.addstr(3, 0, f"Scale: {self._scale}")
        stdscr.addstr(5, 0, "Use arrow keys to move")
        stdscr.addstr(6, 0, "Press 'H' for Home (reset position)")
        stdscr.addstr(7, 0, "Press 'I' to increase scale, 'D' to decrease scale")
        stdscr.addstr(8, 0, "Press 'Q' to quit")
        stdscr.refresh()

        # Limit screen update rate
        curses.napms(1)

        return True

    def home(self):
        self._gantry.set_current_position(0, 0)
        self._target_x = 0
        self._target_y = 0
        self.set_position(0, 0)

    @property
    def x(self):
        return self._target_x

    @x.setter
    def x(self, value):
        delta = value - self._target_x
        self._target_x = value
        self.set_position(delta, 0)

    @property
    def y(self):
        return self._target_y

    @y.setter
    def y(self, value):
        delta = value - self._target_y
        self._target_y = value
        self.set_position(0, delta)

    @property
    def xy(self):
        return self._target_x, self._target_y

    @xy.setter
    def xy(self, xy):
        x, y = xy
        dx, dy = x - self._target_x, y - self._target_y
        self._target_x, self._target_y = x, y
        self.set_position(dx, dy)

    def set_position(self, dx, dy):
        self._gantry.set_position(dx, dy, 0, 0, 0, 0)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Communication script between a PC and the Arduino. "
        "For controlling linear slides with a CNCShield."
    )

    parser.add_argument(
        "-G",
        "--gantry",
        type=str,
        help="The type of gantry to use. Default is DualDrive2AxisGantry.",
        default="DualDrive2AxisGantry",
    )
    parser.add_argument(
        "--port",
        type=str,
        help="The port the arduino is on. If None, auto port detection is used.",
        default=None,
    )
    parser.add_argument(
        "-E" "--exit-immediately",
        dest="exit_immediately",
        action="store_true",
        help="Exit immediately after instantiating the gantry object.",
    )

    args = parser.parse_args()

    jogger = Jogger(args.gantry, args.port)
    if args.exit_immediately:
        return
    jogger.start()


if __name__ == "__main__":
    main()
