from typing import Any
import threading
from collections import deque


class BlockingDeque:
    def __init__(self, *args, **kwargs):
        self._deque = deque(*args, **kwargs)
        self._condition = threading.Condition()

    def append(self, item: Any) -> None:
        with self._condition:
            self._deque.append(item)
            self._condition.notify()

    def __getattr__(self, name: str) -> Any:
        with self._condition:
            return getattr(self._deque, name)

    def __getitem__(self, index: int) -> Any:
        with self._condition:
            while not self._deque:
                self._condition.wait()
            return self._deque[index]

    def __len__(self) -> int:
        with self._condition:
            return len(self._deque)

    def __repr__(self) -> str:
        with self._condition:
            return repr(self._deque)
