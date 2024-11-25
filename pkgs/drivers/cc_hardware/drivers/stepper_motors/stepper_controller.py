"""Stepper motor controller classes."""

from abc import ABC, abstractmethod

import numpy as np

from cc_hardware.utils.registry import Registry, register

# ======================


@register
class StepperController(Registry, ABC):
    @abstractmethod
    def get_position(self, iter: int) -> list[float]:
        """Get the position for the given iteration.

        Args:
            iter (int): The iteration number.

        Returns:
            list[float]: The position for the given iteration.
        """
        pass


# ======================


@register
class SnakeStepperController(StepperController):
    def __init__(self, axis_configs: list[dict]):
        """
        Initialize the controller with a list of axis configurations.

        Args:
            axis_configs (list of dict): A list where each dict represents
                an axis configuration with keys:
                    - 'name' (str): Axis name.
                    - 'range' (tuple): The range (min, max) for the axis.
                    - 'samples' (int): Number of samples along this axis.
        """
        self.axes = []
        self.positions = []
        self.total_positions = 1

        for axis in axis_configs:
            assert "name" in axis, "Axis name is required."
            assert "range" in axis, "Axis range is required."
            assert "samples" in axis, "Number of samples is required."

            axis_name = axis["name"]
            axis_range = axis["range"]
            num_samples = axis["samples"]

            positions = np.linspace(axis_range[0], axis_range[1], num_samples)
            self.axes.append((axis_name, positions))
            self.positions.append(positions)
            self.total_positions *= num_samples

    def get_position(self, iter: int) -> dict:
        """
        Get the current position for all axes.

        Args:
            iter (int): The current iteration index.

        Returns:
            dict: A dictionary with axis names as keys and current positions as values.
                  Returns an empty dictionary if the iteration exceeds total positions.
        """
        if iter >= self.total_positions:
            return {}  # We're done

        current_position = {}
        stride = self.total_positions

        for axis_name, axis_positions in self.axes:
            stride //= len(axis_positions)
            index = (iter // stride) % len(axis_positions)

            if self.axes.index((axis_name, axis_positions)) % 2 == 0:
                # Even index axis: move forward
                current_position[axis_name] = axis_positions[index]
            else:
                # Odd index axis: move backward
                current_position[axis_name] = axis_positions[
                    len(axis_positions) - 1 - index
                ]

        return current_position
