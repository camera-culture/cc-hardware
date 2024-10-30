from abc import ABC, abstractmethod
from typing import Any

from cc_hardware.utils.registry import Registry


class Algorithm(Registry, ABC):
    """This is an algorithm interface that all algorithms should implement."""

    def __init__(self):
        pass

    @abstractmethod
    def run(self) -> Any:
        """Runs the algorithm and returns the result.

        Each subclass can add additional parameters to this method and specify it's
        return type.
        """
        pass

    @property
    @abstractmethod
    def is_okay(self) -> bool:
        """Returns True if the algorithm is okay to run, False otherwise.
        An algorithm may not be okay if it either has not been initialized properly or
        if it has encountered an error.
        """
        pass

    def close(self):
        """Closes the algorithm and releases any resources. A default implementation
        is provided here, but subclasses can override this method to provide their own
        implementation.
        """
        pass
