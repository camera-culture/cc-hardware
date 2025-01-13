"""
This module provides a thread-safe wrapper around a deque
object, ensuring synchronized access and blocking behavior
when attempting to retrieve items from an empty deque.
"""

import threading
from collections import deque
from typing import Any


class BlockingDeque:
    """
    A thread-safe deque wrapper with blocking behavior for item retrieval.

    This class wraps a deque and synchronizes access using threading.Condition,
    making it suitable for use in multi-threaded environments where safe
    access to a shared deque is required.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the BlockingDeque.

        Arguments are passed to the underlying deque's constructor.

        Args:
            *args: Positional arguments for the deque constructor.
            **kwargs: Keyword arguments for the deque constructor.
        """
        self._deque = deque(*args, **kwargs)
        self._condition = threading.Condition()

    def append(self, item: Any) -> None:
        """
        Append an item to the deque and notify any waiting threads.

        Args:
            item (Any): The item to append to the deque.
        """
        with self._condition:
            self._deque.append(item)
            self._condition.notify()

    def __getattr__(self, name: str) -> Any:
        """
        Access attributes of the underlying deque in a thread-safe manner.

        Args:
            name (str): The name of the attribute to access.

        Returns:
            Any: The value of the requested attribute.
        """
        with self._condition:
            return getattr(self._deque, name)

    def __getitem__(self, index: int) -> Any:
        """
        Retrieve an item from the deque by index, blocking if the deque is empty.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Any: The item at the specified index.
        """
        with self._condition:
            while not self._deque:
                self._condition.wait()
            return self._deque[index]

    def __len__(self) -> int:
        """
        Get the number of items in the deque.

        Returns:
            int: The number of items in the deque.
        """
        with self._condition:
            return len(self._deque)

    def __repr__(self) -> str:
        """
        Get the string representation of the deque.

        Returns:
            str: The string representation of the deque.
        """
        with self._condition:
            return repr(self._deque)
