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


class Manager:
    """This is a manager for handling components which must be closed. It is
    essentially just a context manager which calls close on all components when
    it is closed.
    """

    def __init__(self, **components: Type[Component] | Component):
        self._components = components

        self._closed = False

    def add(self, **components: Component):
        """Adds additional components to the manager."""
        # Check each component has a close method
        for name, component in components.items():
            if not hasattr(component, "close"):
                get_logger().warning(f"Component {name} does not have a close method.")
                component.close = lambda: None
            if not hasattr(component, "is_okay"):
                get_logger().warning(f"Component {name} does not have an is_okay prop.")
                component.is_okay = True

        self._components.update(components)

    def run(
        self,
        *,
        setup: Callable[..., None] | None = None,
        loop: Callable[..., bool] | None = None,
        cleanup: Callable[..., None] | None = None,
    ) -> None:
        """Runs a setup and loop function until all components are okay.

        Args:
            setup (Callable[..., None] | None, optional): Setup function to run before
                the loop. Accepts keyword arguments, returns None. Defaults to None.
            loop (Callable[..., bool] | None, optional): Loop function to run until all
                components are okay. Accepts keyword arguments, returns bool. When
                False, the loop will stop and cleanup will begin. Defaults to None.
            cleanup (Callable[..., None] | None, optional): Cleanup function to run
                after the loop. Accepts keyword arguments, returns None. Defaults to
                None.
        """
        setup = setup or (lambda **_: None)
        loop = loop or (lambda *_, **__: None)
        cleanup = cleanup or (lambda **_: None)

        try:
            setup(manager=self, **self._components)
        except Exception:
            get_logger().exception("Failed to setup components.")
            self.close()
            return

        i = 0
        while self.is_okay:
            try:
                if not loop(i, manager=self, **self._components):
                    break
            except Exception:
                get_logger().exception(f"Failed to run loop {i}.")
                self.close()
                break

            i += 1

        try:
            cleanup(manager=self, **self._components)
        except Exception:
            get_logger().exception("Failed to cleanup components.")
            self.close()
            return

        self.close()

    def __enter__(self):
        """Allows this class to be used as a context manager."""
        # Create each component if it is a type
        for name, component in self._components.items():
            try:
                if isinstance(component, type):
                    self._components[name] = component()
            except Exception:
                get_logger().exception(f"Failed to create component {name}.")
                self.close()
                raise

        # Check each component has a close method
        for name, component in self._components.items():
            if not hasattr(component, "close"):
                get_logger().warning(f"Component {name} does not have a close method.")
                component.close = lambda: None
            if not hasattr(component, "is_okay"):
                get_logger().warning(f"Component {name} does not have an is_okay prop.")
                component.is_okay = True

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensures each component is properly closed when used as a context manager."""
        self.close()

    @property
    def components(self) -> dict[str, Component]:
        """Returns a dictionary of components."""
        return self._components

    @property
    def is_okay(self) -> bool:
        """Checks if all components are okay."""
        return all(component.is_okay for component in self._components.values())

    def close(self):
        """Closes all components."""
        if self._closed:
            return

        for name, component in self._components.items():
            if isinstance(component, type):
                continue

            get_logger().info(f"Closing {name}...")
            try:
                component.close()
            except Exception:
                get_logger().exception(
                    f"Failed to close {name} ({component.__class__.__name__})."
                )

        self._closed = True
