"""Registry base class and decorator for registering classes in the registry."""

from enum import Enum
from typing import Any, Self

from cc_hardware.utils import classproperty


class Registry:
    """
    A base class that provides a registry for its subclasses and a factory method
    to instantiate them.
    """

    _registry: dict[str, dict[str, Self]] = {}

    @classmethod
    def register(cls: type[Self], class_type: type[Self]) -> type[Self]:
        """
        Register the given class in the registry of the base class.

        Args:
            class_type: The class to be registered.

        Returns:
            The registered class.
        """
        cls._registry.setdefault(cls.__name__, {})[class_type.__name__] = class_type
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
        if cls.__name__ not in cls._registry:
            raise ValueError(
                f"Class '{name}' not found in {cls.__name__}'s registry. "
                "Ensure the class inherits from Registry."
            )
        elif name not in cls._registry[cls.__name__]:
            raise ValueError(
                f"Class '{name}' not found in {cls.__name__}'s registry. "
                f"Available classes: {list(cls._registry[cls.__name__].keys())}"
            )
        class_type = cls._registry[cls.__name__][name]
        return class_type(*args, **kwargs)

    @classproperty
    def registry(cls: type[Self]) -> dict[str, Self]:
        """
        Get the registry of the base class.

        Returns:
            The registry of the base class.
        """
        return cls._registry.get(cls.__name__, {})

    @classproperty
    def registered(cls: type[Self]) -> Enum:
        """
        Get an enumeration of the registered classes.

        Returns:
            An enumeration of the registered classes.
        """
        return Enum(cls.__name__, {name: name for name in cls.registry})


def register(class_type: type[Registry]) -> type[Registry]:
    """
    Register the given class using the RegistryBase of its base class.

    Args:
        class_type: The class to register.

    Returns:
        The registered class.
    """

    def register_with_bases(cls: type[Registry]):
        for base in cls.__bases__:
            if issubclass(base, Registry):
                base.register(class_type)
                register_with_bases(base)

    # Register the class in the registry of each of its base classes that are
    # subclasses of RegistryBase
    register_with_bases(class_type)

    return class_type
