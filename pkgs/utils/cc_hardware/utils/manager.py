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

from functools import partial
from typing import Callable, Protocol, Type

from cc_hardware.utils.logger import get_logger


class Component(Protocol):
    """Protocol for components which can be closed."""

    def close(self) -> None:
        """Closes the component and releases any resources."""
        ...

    @property
    def is_okay(self) -> bool:
        """Checks if the component is operational."""
        ...


class PrimitiveComponent(Component):
    """Wrapper for primitive components which do not need to be closed."""

    def __init__(self, value):
        self._value = value

    def close(self) -> None:
        pass

    @property
    def is_okay(self) -> bool:
        return True

    @property
    def value(self):
        return self._value


class Manager:
    """This is a manager for handling components which must be closed. It is
    essentially just a context manager which calls close on all components when
    it is closed.
    """

    def __init__(self, **components: Type[Component] | Component):
        self._components = components

        self._closed = False

    def add(self, *, primitive: bool = False, **components: Component):
        """Adds additional components to the manager."""
        # Check each component has a close method
        _components = {}
        for name, component in components.items():
            if component is None:
                _components[name] = component
                continue

            if primitive:
                component = PrimitiveComponent(component)

            if not hasattr(component, "close"):
                get_logger().warning(f"Component {name} does not have a close method.")
                component.close = lambda: None
            if not hasattr(component, "is_okay"):
                get_logger().warning(f"Component {name} does not have an is_okay prop.")
                component.is_okay = True

            _components[name] = component

        self._components.update(_components)

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
        setup = setup or (lambda **_: None)
        loop = loop or (lambda *_, **__: None)
        cleanup = cleanup or (lambda **_: None)

        def get_components():
            return {
                name: (
                    component.value
                    if isinstance(component, PrimitiveComponent)
                    else component
                )
                for name, component in self._components.items()
            }

        try:
            # SETUP
            try:
                if setup(manager=self, **get_components()) == False:
                    get_logger().info("Exiting setup.")
                    return
            except Exception:
                get_logger().exception("Failed to setup components.")
                return

            # LOOP
            while self.is_okay:
                try:
                    if loop(iter, manager=self, **get_components()) == False:
                        get_logger().info(f"Exiting loop after {iter} iterations.")
                        break
                except Exception:
                    get_logger().exception(f"Failed to run loop {iter}.")
                    break

                iter += 1

            # CLEANUP
            try:
                cleanup(manager=self, **get_components())
            except Exception:
                get_logger().exception("Failed to cleanup components.")
                return
        finally:
            self.close()

    def __enter__(self):
        """Allows this class to be used as a context manager."""
        # Create each component if it is a type
        for name, component in self._components.items():
            if component is None:
                continue

            try:
                if isinstance(component, (type, partial)):
                    self._components[name] = component()
            except Exception:
                get_logger().exception(f"Failed to create component {name}.")
                self.close()
                raise

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensures each component is properly closed when used as a context manager."""
        self.close()

    def get(self, name: str) -> Component | None:
        """Returns a component by name. Alias to :meth:`get_component`."""
        return self.get_component(name)

    def get_component(self, name: str) -> Component | None:
        """Returns a component by name."""
        if name not in self._components:
            get_logger().error(f"Component {name} not found.")
            return None
        return self._components[name]

    @property
    def components(self) -> dict[str, Component]:
        """Returns a dictionary of components."""
        return self._components

    @property
    def is_okay(self) -> bool:
        """Checks if all components are okay."""
        for name, component in self._components.items():
            if component is None:
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
            if isinstance(component, (type, partial)):
                continue
            elif component is None:
                continue

            get_logger().info(f"Closing {name}...")
            try:
                component.close()
            except Exception:
                get_logger().exception(
                    f"Failed to close {name} ({component.__class__.__name__})."
                )

        self._closed = True
