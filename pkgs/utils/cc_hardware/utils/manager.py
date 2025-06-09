"""This module contains a manager for handling components which must be closed.

The manager can be used as a context manager or run with a setup, loop, and cleanup
function. The manager will ensure all components are properly closed when it is closed.

Example:

.. code-block:: python

    from cc_hardware.utils.manager import Manager

    def setup(manager, camera):
        camera.start()

    def loop(i, manager, camera):
        if i > 100:
            return False
        return camera.is_okay

    def cleanup(manager, camera):
        camera.stop()

    with Manager(camera=Camera()) as manager:
        manager.run(setup=setup, loop=loop, cleanup=cleanup)
"""

import queue
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future
from functools import partial
from typing import Any, Callable, Self, TypeVar, cast

from hydra_config import HydraContainerConfig, config_wrapper

from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.registry import Registry


@config_wrapper
class Config(ABC, HydraContainerConfig, Registry):
    """Base configuration class for cc-hardware components."""

    pass


class Component[T: Config](ABC, Registry):
    """Base class for components which must be closed."""

    def __init__(self, config: T):
        self._config = config

    @property
    def config(self) -> T:
        """Retrieves the component configuration."""
        return self._config

    @classmethod
    def create_from_config(cls, config: T, **kwargs) -> Self:
        """Create an instance of the class from a configuration object.

        Args:
            config (T): The configuration object.

        Returns:
            Self: An instance of the class.
        """
        return config.create_from_registry(config=config, **kwargs)

    # ==================

    @abstractmethod
    def close(self) -> None:
        """Closes the component and releases any resources."""
        ...

    @property
    @abstractmethod
    def is_okay(self) -> bool:
        """Checks if the component is operational."""
        ...

    def __del__(self):
        """Ensures the component is closed when it is deleted."""
        self.close()


class ThreadedComponent[T: Config](Component[T]):
    def __init__(self, wrapped: Component[T], *, future: bool = True):
        # avoid __setattr__ before internals exist
        self._thread: threading.Thread
        self._tasks: queue.Queue
        self._future: bool
        self._wrapped: Component[T]
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_future", future)
        object.__setattr__(self, "_tasks", queue.Queue())
        object.__setattr__(
            self, "_thread", threading.Thread(target=self._worker, daemon=True)
        )

        # now safe
        super().__init__(wrapped.config)

        self._thread.start()

    def __getattr__(self, name: str):
        attr = getattr(self._wrapped, name)
        if callable(attr):

            def _call(*args, **kwargs):
                fut = Future()
                self._tasks.put((attr, args, kwargs, fut))
                return fut if self._future else fut.result()

            return _call
        return self._sync(lambda: getattr(self._wrapped, name))

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_") or "_tasks" not in self.__dict__:
            object.__setattr__(self, name, value)
        else:
            self._sync(lambda: setattr(self._wrapped, name, value))

    @property
    def is_okay(self) -> bool:
        return self._thread.is_alive() and self._wrapped.is_okay

    def close(self):
        if not self._thread.is_alive():
            return

        self._sync(self._wrapped.close)
        self._tasks.put(None)
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            get_logger().warning(
                f"Could not close {self._wrapped.__class__.__name__} gracefully, "
            )

    def _sync(self, fn: Callable, *, force_result: bool = False) -> Any | Future:
        fut = Future()
        self._tasks.put((fn, (), {}, fut))
        if force_result or not self._future:
            return fut.result()
        return fut

    def _worker(self):
        while True:
            item = self._tasks.get()
            if item is None:
                break
            fn, args, kwargs, fut = item
            try:
                res = fn(*args, **kwargs)
                if fut:
                    fut.set_result(res)
            except Exception as e:
                if fut:
                    fut.set_exception(e)


_C = TypeVar("_C", bound=Component)


def threaded_component(component: _C, **kwargs) -> _C | ThreadedComponent:
    return cast(_C | ThreadedComponent, ThreadedComponent(component, **kwargs))


class Manager:
    """This is a manager for handling components which must be closed. It is
    essentially just a context manager which calls close on all components when
    it is closed.
    """

    def __init__(
        self,
        *,
        cleanup_on_keyboard_interrupt: bool = True,
        **components: type[Component] | Component | Any,
    ):
        self._components: dict[type[Component] | Component | Any] = components
        self._cleanup_on_keyboard_interrupt = cleanup_on_keyboard_interrupt

        self._closed = False
        self._looping = True  # TODO: shouldn't start as true

    def add(self, **components: Component | Any):
        """Adds additional components to the manager."""
        # Check each component has a close method
        self._components.update(components)

    def run(
        self,
        iter: int = 0,
        *,
        setup: Callable[..., bool | None] | None = None,
        loop: Callable[..., bool | None] | None = None,
        cleanup: Callable[..., None] | None = None,
    ) -> None:
        """Runs a setup and loop function until all components are okay.

        Args:
            iter (int, optional): Iteration counter. Defaults to 0.

        Keyword Args:
            setup (Callable[..., bool | None] | None, optional): Setup function to run
                before the loop. Accepts keyword arguments, returns None. Defaults to
                None. A return value of False will stop the run.
            loop (Callable[..., bool | None] | None, optional): Loop function to run
                until all components are okay. Accepts keyword arguments, returns bool.
                When False, the loop will stop and cleanup will begin. Defaults to None.
                A return value of None will continue the loop.
            cleanup (Callable[..., None] | None, optional): Cleanup function to run
                after the loop. Accepts keyword arguments, returns None. Defaults to
                None.
        """
        self._iter = iter

        setup = setup or (lambda **_: None)
        loop = loop or (lambda *_, **__: None)
        cleanup = cleanup or (lambda **_: None)

        try:
            # SETUP
            try:
                if setup(manager=self, **self._components) is False:
                    get_logger().info("Setup failed, exiting...")
                    return
            except Exception:
                get_logger().exception("Failed to setup components.")
                return

            # LOOP
            self._looping = True
            try:
                while self.is_okay:
                    if loop(self._iter, manager=self, **self._components) is False:
                        get_logger().info(f"Exiting loop after {self._iter + 1} iterations.")
                        break

                    self._iter += 1
            except KeyboardInterrupt:
                if not self._cleanup_on_keyboard_interrupt:
                    get_logger().info("Exiting loop.")
                    return
            except Exception as e:
                get_logger().exception(f"Failed to run loop {iter}: {e}")
                return
            finally:
                self._looping = False

            # CLEANUP
            try:
                get_logger().info("Cleaning up...")
                cleanup(manager=self, **self._components)
            except Exception:
                get_logger().exception("Failed to cleanup components.")
                return
        finally:
            # Ensure all components are closed
            self.close()

    def __enter__(self):
        """Allows this class to be used as a context manager."""
        # Create each component if it is a type
        for name, component in self._components.items():
            if not isinstance(component, (type, partial)):
                continue

            try:
                if isinstance(component, (type, partial)):
                    self._components[name] = component()
            except Exception:
                get_logger().exception(f"Failed to create component {name}.")
                self.close()
                raise

        return self

    def __exit__(self, *_, **__):
        """Ensures each component is properly closed when used as a context manager."""
        self.close()

    def set_iter(self, iter: int):
        """Sets the iteration counter."""
        self._iter = iter

    @property
    def components(self) -> dict[str, type[Component] | Component | Any]:
        """Returns a dictionary of components."""
        return self._components

    @property
    def is_looping(self) -> bool:
        """Checks if the manager is currently looping."""
        return self._looping

    @property
    def is_okay(self) -> bool:
        """Checks if all components are okay."""
        for name, component in self._components.items():
            if not isinstance(component, Component):
                continue

            if not component.is_okay:
                get_logger().error(f"Component {name} is not okay.")
                return False
        return True

    def close(self):
        """Closes all components."""
        if self._closed:
            return

        for name, component in self._components.items():
            if not isinstance(component, Component):
                continue

            get_logger().info(f"Closing {name}...")
            try:
                component.close()
            except Exception as e:
                get_logger().exception(
                    f"Failed to close {name} ({component.__class__.__name__}): {e}"
                )

        self._closed = True
