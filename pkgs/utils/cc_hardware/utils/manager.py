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
        # Check each component has a close method
        for name, component in components.items():
            if not hasattr(component, "close"):
                get_logger().warning(f"Component {name} does not have a close method.")
                component.close = lambda: None
            if not hasattr(component, "is_okay"):
                get_logger().warning(f"Component {name} does not have an is_okay prop.")
                component.is_okay = True

        self._components = components

    def add(self, **components: Component):
        """Adds additional components to the manager."""
        self._components.update(components)

    def run(
        self,
        *,
        setup: Callable[..., None] | None = None,
        loop: Callable[..., None] | None = None,
    ) -> None:
        """Runs a setup and loop function until all components are okay.

        Args:
            setup (Callable[..., None] | None, optional): Setup function to run before
                the loop. Accepts keyword arguments, returns None. Defaults to None.
            loop (Callable[..., None] | None, optional): Loop function to run until all
                components are okay. Accepts keyword arguments, returns None.
                Defaults to None.
        """
        setup = setup or (lambda **_: None)
        loop = loop or (lambda **_: None)

        setup(manager=self, **self._components)
        while self.is_okay:
            loop(manager=self, **self._components)

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
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensures each component is properly closed when used as a context manager."""
        self.close()

    def __getattr__(self, name):
        """Allows components to be accessed as attributes."""
        if name in self._components:
            return self._components[name]
        return super().__getattr__(name)

    @property
    def is_okay(self) -> bool:
        """Checks if all components are okay."""
        return all(component.is_okay for component in self._components.values())

    def close(self):
        """Closes all components."""
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
