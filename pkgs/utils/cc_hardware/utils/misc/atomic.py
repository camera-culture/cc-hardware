import threading
from typing import Any


class AtomicVariable:
    def __init__(self, value: Any):
        self._value = value
        self._lock = threading.Lock()

    def get(self) -> Any:
        with self._lock:
            return self._value

    def set(self, value: Any):
        with self._lock:
            self._value = value
