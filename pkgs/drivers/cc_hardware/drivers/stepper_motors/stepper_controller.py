"""Stepper motor controller classes."""

from abc import ABC, abstractmethod

import numpy as np

from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.registry import Registry, register

# ======================


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
        self.axes = {}
        self.total_positions = 1

        for axis_index, axis in enumerate(axis_configs):
            assert "name" in axis, "Axis name is required."
            assert "range" in axis, "Axis range is required."
            assert "samples" in axis, "Number of samples is required."

            axis_name = axis["name"]
            axis_range = axis["range"]
            num_samples = axis["samples"]

            positions = np.linspace(axis_range[0], axis_range[1], num_samples)
            self.axes[axis_name] = dict(
                name=axis_name, index=axis_index, positions=positions, flipped=True
            )
            self.total_positions *= num_samples

    def get_position(self, iter: int) -> dict | None:
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

        for axis in self.axes.values():
            stride //= len(axis["positions"])
            if iter % len(axis["positions"]) == 0:
                axis["flipped"] = not axis["flipped"]
            index = (iter // stride) % len(axis["positions"])
            reverse_index = len(axis["positions"]) - 1 - index

            if axis["index"] % 2 == 0:
                current_position[axis["name"]] = axis["positions"][index]
            else:
                index = reverse_index if axis["flipped"] else index
                current_position[axis["name"]] = axis["positions"][index]

            get_logger().info(f"Axis {axis['name']}: {current_position[axis['name']]}")

        return current_position
