"""Stepper motor controller classes."""

from abc import abstractmethod
from dataclasses import field

import numpy as np

from cc_hardware.utils import Component, Config, config_wrapper, get_logger

# ======================


@config_wrapper
class ControllerAxisConfig(Config):
    """Configuration for a controller axis.

    Attributes:
        index (int): The index of the axis.
        positions (list[float]): A list of positions for the axis.
        flipped (bool): Whether the axis is flipped or not.
    """

    index: int
    positions: list[float]
    flipped: bool = False


@config_wrapper
class StepperControllerConfig(Config):
    """Configuration for a stepper motor controller."""

    pass


class StepperController[T: StepperControllerConfig](Component[T]):
    def __init__(self, config: T):
        """
        Initialize the stepper controller with the given configuration.

        Args:
            config (T): The configuration for the stepper controller.
        """
        super().__init__(config)

        self.total_positions = 1

    @abstractmethod
    def get_position(self, iter: int, *, verbose: bool = True) -> list[float]:
        """Get the position for the given iteration.

        Args:
            iter (int): The iteration number.

        Returns:
            list[float]: The position for the given iteration.
        """
        pass

    @property
    def is_okay(self) -> bool:
        """Check if the controller is okay.

        Returns:
            bool: True if the controller is okay, False otherwise.
        """
        return True

    def close(self):
        """Close the controller."""
        pass


# ======================


@config_wrapper
class SnakeControllerAxisConfig(Config):
    """Configuration for a single axis of a stepper motor controller.

    Attributes:
        range (tuple): The range (min, max) for the axis.
        samples (int): Number of samples along this axis.
    """

    range: tuple[float, float]
    samples: int


@config_wrapper
class SnakeStepperControllerConfig(StepperControllerConfig):
    """Configuration for a stepper motor controller.

    Attributes:
        axes (dict[str, ControllerAxisConfig]): A list of axis configurations. Note
            that the keys must be unique and are used to identify the axes in the
            controller.
    """

    axes: dict[str, SnakeControllerAxisConfig]


@config_wrapper
class SnakeStepperControllerConfigXY(SnakeStepperControllerConfig):
    """Configuration for a stepper motor controller with 2 axes.

    Attributes:
        axes (dict[str, ControllerAxisConfig]): A list of axis configurations. Note
            that the keys must be unique and are used to identify the axes in the
            controller.
    """

    axes: dict[str, SnakeControllerAxisConfig] = field(
        default_factory=lambda: {
            "x": SnakeControllerAxisConfig(range=(0, 16), samples=10),
            "y": SnakeControllerAxisConfig(range=(0, 16), samples=10),
        }
    )


class SnakeStepperController(StepperController[StepperControllerConfig]):
    def __init__(self, config: SnakeStepperControllerConfig):
        """
        Initialize the controller with a list of axis configurations.

        Args:
            config (T): The configuration for the stepper controller.
        """

        super().__init__(config)

        axes = {}
        for axis_index, (name, axis) in enumerate(config.axes.items()):
            positions = np.linspace(axis.range[0], axis.range[1], axis.samples)
            axes[name] = ControllerAxisConfig(
                index=axis_index, positions=positions, flipped=True
            )
            self.total_positions *= axis.samples
        self.axes: dict[str, ControllerAxisConfig] = axes

    def get_position(self, iter: int, *, verbose: bool = True) -> dict | None:
        """
        Get the position that the controller should move to for the given iteration.

        Args:
            iter (int): The current iteration index.

        Returns:
            dict: A dictionary with axis names as keys and current positions as values.
                  Returns an empty dictionary if the iteration exceeds total positions.
        """
        if iter >= self.total_positions:
            return None

        current_position = {}
        stride = self.total_positions

        for name, axis in self.axes.items():
            stride //= len(axis.positions)
            if iter % len(axis.positions) == 0:
                axis.flipped = not axis.flipped
            index = (iter // stride) % len(axis.positions)
            reverse_index = len(axis.positions) - 1 - index
            if axis.index % 2 == 0:
                current_position[name] = axis.positions[index]
            else:
                index = reverse_index if axis.flipped else index
                current_position[name] = axis.positions[index]
            if verbose:
                get_logger().info(f"Axis {name}: {current_position[name]}")
            else:
                get_logger().debug(f"Axis {name}: {current_position[name]}")

        return current_position
