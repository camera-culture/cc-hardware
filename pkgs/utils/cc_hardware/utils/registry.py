"""Registry base class and decorator for registering classes in the registry."""

from typing import Any, Self


class Registry:
    """
    A base class that provides a registry for its subclasses and a factory method
    to instantiate them.
    """

    registry: dict[str, dict[str, Self]] = {}

    @classmethod
    def register(cls: type[Self], class_type: type[Self]) -> type[Self]:
        """
        Register the given class in the registry of the base class.

        Args:
            class_type: The class to be registered.

        Returns:
            The registered class.
        """
        cls.registry.setdefault(cls.__name__, {})[class_type.__name__] = class_type
        return class_type

    @classmethod
    def create_from_registry(
        cls: type[Self], name: str, *args: Any, **kwargs: Any
    ) -> Self:
        """
        Create an instance of the class with the specified name from the registry.

        Args:
            name: The name of the class to instantiate.
            *args: Positional arguments to pass to the class constructor.
            **kwargs: Keyword arguments to pass to the class constructor.

        Returns:
            An instance of the class.

        Raises:
            ValueError: If the class is not found in the registry.
        """
        if cls.__name__ not in cls.registry:
            raise ValueError(
                f"Class '{name}' not found in {cls.__name__}'s registry. "
                "Ensure the class inherits from Registry."
            )
        elif name not in cls.registry[cls.__name__]:
            raise ValueError(
                f"Class '{name}' not found in {cls.__name__}'s registry. "
                f"Available classes: {list(cls.registry[cls.__name__].keys())}"
            )
        class_type = cls.registry[cls.__name__][name]
        return class_type(*args, **kwargs)


def register(class_type: type[Registry]) -> type[Registry]:
    """
    Register the given class using the RegistryBase of its base class.

    Args:
        class_type: The class to register.

    Returns:
        The registered class.
    """
    # Register the class in the registry of each of its base classes that are
    # subclasses of RegistryBase
    for base in class_type.__bases__:
        if issubclass(base, Registry):
            base.register(class_type)
    return class_type
