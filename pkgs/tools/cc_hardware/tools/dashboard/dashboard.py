from typing import Callable, Self
from abc import abstractmethod

from cc_hardware.utils import Component, Config, config_wrapper, get_logger


@config_wrapper
class DashboardConfig(Config):
    """Configuration for dashboard GUIs.

    Attributes:
        num_frames (int): Number of frames to process. Default is 1,000,000.
        user_callback (Callable[[Self], None], optional): User-defined callback
            function. It should accept the dashboard instance as an argument.
    """

    num_frames: int = 1_000_000
    user_callback: Callable[[Self], None] | None = None


class Dashboard[T: DashboardConfig](Component[T]):
    """
    Abstract base class for dashboards.

    Args:
        config (DashboardConfig): The dashboard configuration
    """

    def __init__(self, config: DashboardConfig):
        super().__init__(config)

    @abstractmethod
    def setup(self):
        """
        Abstract method to set up the dashboard. Should be independent of whether the
        dashboard is run in a loop or not.
        """
        pass

    @abstractmethod
    def run(self):
        """
        Abstract method to display the dashboard. Blocks until the dashboard is closed.
        """
        pass

    def update(self, frame: int):
        """
        Method to update dashboard synchronously. This should be capable of being
        used independent of the loop, as in a main thread and non-blocking.

        Args:
            frame (int): Current frame number.
        """
        raise NotImplementedError
