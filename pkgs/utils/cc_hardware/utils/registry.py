"""Registry base class and decorator for registering classes in the registry."""

from enum import Enum
from typing import Any, Self, overload

from cc_hardware.utils import classproperty, get_object


class Registry:
    """
    A base class that provides a registry for its subclasses and a factory method
    to instantiate them.

    Supports both direct and lazy registration of classes, allowing classes
    to be registered by name and module path for deferred loading. An enumeration
    of registered classes is also provided.
    """

    _registry: dict[str, dict[str, Any]] = {}

    @overload
    @classmethod
    def register(cls: type[Self], class_type: type[Self]) -> type[Self]:
        """
        Register the given class in this registry.

        Args:
            class_type: The class to register.

        Returns:
            The registered class.
        """
        ...

    @overload
    @classmethod
    def register(cls: type[Self], class_name: str, module_path: str) -> None:
        """
        Register a class lazily by specifying its module path instead of
        importing it directly.

        Args:
            class_name: The name of the class to register.
            module_path: The module path where the class can be found
                (e.g. "my_module.MyClass").
        """
        ...

    @classmethod
    def register(
        cls: type[Self], class_type: type[Self] | str, module_path: str | None = None
    ) -> type[Self] | None:
        """
        Register the given class in this registry.

        Args:
            class_type: The class to register. Can be a class type or a str defining
                 a class name to be lazily loaded. If the class_type is a type, this
                 method will return the class itself (useful for decorators).
            module_path: The module path where the class can be found
                (e.g. "my_module.my_submodule"). Only used if class_type is a str.
                This is the full importable path to the class. The final class
                path will be ``module_path.class_type``.

        Returns:
            The registered class if class_type is a class type, otherwise None.
        """
        if isinstance(class_type, type):
            cls._registry.setdefault(cls.__name__, {})[class_type.__name__] = class_type
            return class_type
        elif isinstance(class_type, str):
            assert (
                module_path is not None
            ), "module_path must be provided for lazy loading."
            module_path = f"{module_path}.{class_type}"
            cls._registry.setdefault(cls.__name__, {})[class_type] = module_path
        else:
            raise ValueError(
                f"Invalid class type: {class_type}. Must be a class type or a string."
            )
        return None

    @classmethod
    def create_from_registry(
        cls: type[Self], name: str, *args: Any, **kwargs: Any
    ) -> Self:
        """
        Create an instance of a registered class, performing lazy loading if necessary.

        Args:
            name: The name of the class to instantiate.
            *args: Positional arguments to pass to the class constructor.
            **kwargs: Keyword arguments to pass to the class constructor.

        Returns:
            An instance of the requested class.

        Raises:
            ValueError: If the class name is not registered.
        """
        if cls.__name__ not in cls._registry:
            raise ValueError(f"Class '{name}' not found in {cls.__name__}'s registry.")
        val = cls._registry[cls.__name__].get(name, None)
        if val is None:
            raise ValueError(f"Class '{name}' not found in {cls.__name__}'s registry.")

        # Lazy loading logic
        if isinstance(val, str):
            # 'val' is a module path like "my_module.MyClass"
            class_type = get_object(val)
            # Store the fully loaded class back into the registry for future calls
            cls._registry[cls.__name__][name] = class_type
        else:
            class_type = val

        return class_type(*args, **kwargs)

    @classproperty
    def registry(cls: type[Self]) -> dict[str, Any]:
        """
        Get the registry for this class.

        Returns:
            A dictionary mapping class names to class objects or lazy load paths.
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


def register(class_type: type[Registry] | str) -> type[Registry]:
    """
    Decorator to register a class with its base Registry class. Uses a recursive
    approach to ensure that the class is registered with all Registry-based ancestors.

    Args:
        class_type: The class to register.

    Returns:
        The registered class.
    """

    def register_with_bases(cls: type[Registry]):
        # Recursively traverse the class hierarchy, registering the class with each
        # ancestor that is a subclass of Registry.
        for base in cls.__bases__:
            if issubclass(base, Registry) and base is not Registry:
                base.register(class_type)
                register_with_bases(base)

    register_with_bases(class_type)
    return class_type
