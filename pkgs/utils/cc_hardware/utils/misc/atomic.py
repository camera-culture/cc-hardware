import multiprocessing
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


class MPAtomicVariable:
    _shared_manager = None
    _init_lock = multiprocessing.Lock()

    def __init__(self, value: Any = None):
        self._value = None
        self._lock = None
        self._initial = value
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        with MPAtomicVariable._init_lock:
            if MPAtomicVariable._shared_manager is None:
                MPAtomicVariable._shared_manager = multiprocessing.Manager()
        manager = MPAtomicVariable._shared_manager
        self._lock = manager.Lock()
        self._value = manager.Namespace()
        self._value.data = self._initial
        self._initialized = True

    def get(self) -> Any:
        self._ensure_initialized()
        with self._lock:
            return self._value.data

    def set(self, value: Any):
        self._ensure_initialized()
        with self._lock:
            self._value.data = value
