"""
A module that provides a base Registry class to register and instantiate classes.

Example:
    from my_registry import Registry, register

    @register
    class MyClass(Registry):
        def __init__(self, value):
            self.value = value

    # Create instance
    instance = MyClass.create_from_registry('MyClass', 'hello')
    print(instance.value)  # 'hello'

    # Register lazily
    MyClass.register("MyOtherClass", "path.to.my_module")

    # Create instance lazily
    other_instance = MyClass.create_from_registry('MyOtherClass', some_arg='value')
    print(other_instance)
"""

from enum import Enum
from typing import Any, Self, Type, overload

from cc_hardware.utils.logger import get_logger
from cc_hardware.utils.misc import classproperty, get_object


class Registry:
    """Base class providing a registry for its subclasses, plus a factory method.

    This class supports both direct and lazy registration of subclasses and
    can also associate a registered class with a 'friend' (another class not
    necessarily inheriting from Registry).

    Examples:
        Register directly:
            @MyRegistry.register
            class Foo(MyRegistry):
                pass

        Register by name (lazy):
            MyRegistry.register("Bar", "my_module.submodule")

        Create instance:
            instance = MyRegistry.create_from_registry("Foo")

        Create instance from one of many registries:
            instance = MyRegistry.create_from_registry("Bar")
    """

    _registry: dict[str, dict[str, Type[Any]]] = {}

    @overload
    @classmethod
    def register(cls: type[Self], class_type: type[Self]) -> type[Self]:
        """Registers a class, returning the class itself."""
        ...

    @overload
    @classmethod
    def register(cls: type[Self], class_name: str, module_path: str) -> None:
        """Registers a class by name, loading from its module path."""
        ...

    @overload
    @classmethod
    def register(
        cls: type[Self], class_name: str, module_path: str, friend: str
    ) -> None:
        """Associates a class with another 'friend' class in the registry."""
        ...

    @classmethod
    def register(
        cls: type[Self],
        class_type: type[Self] | str,
        module_path: str | None = None,
        friend: str | None = None,
    ) -> type[Self] | type[Any] | None:
        """
        Registers a class in this registry.

        Args:
            class_type (type[Self] | str): A class object or a class name for lazy
                loading.
            module_path (str | None): Path to the module if `class_type` is a string.
            friend (str | None): Name of another class to associate with this class.

        Returns:
            type[Self] | None: The registered class if `class_type` is a type,
                otherwise None.
        """
        if isinstance(class_type, str):
            assert (
                module_path is not None
            ), "module_path must be provided for lazy loading."
            class_module_path = f"{module_path}.{class_type}"
            try:
                class_type = get_object(class_module_path, verbose=False)
            except ImportError as e:
                get_logger().debug(f"Failed to import {class_type}: {e}")
                return None

        name = class_type.__name__
        if friend is not None:
            assert name in cls.registry, (
                f"Class '{name}' must be registered before associating with "
                f"friend '{friend}'."
            )
            assert issubclass(cls.registry[name], Registry), (
                f"Class '{name}' must be a subclass of Registry to be associated "
                f"with friend '{friend}'."
            )
            cls.registry[name].register(friend, module_path)
        else:
            cls.registry[name] = class_type
        return class_type

    @classmethod
    def create_from_registry(
        cls: type[Self], name: str | None = None, *args: Any, **kwargs: Any
    ) -> Self | Any:
        """
        Creates an instance of a registered class, performing lazy loading if necessary.

        Args:
            name (str | None): The name of the class to instantiate. If None,
                and only one class is registered, it will instantiate that class.
            *args (Any): Positional arguments to pass to the class constructor.
            **kwargs (Any): Keyword arguments to pass to the class constructor.

        Returns:
            Self | Any: An instance of the requested class, or an associated
                'friend' class.
        """
        if len(cls.registry) == 0:
            raise ValueError(f"No class registered with {cls.__name__}.")

        if name is None:
            if len(cls.registry) > 1:
                raise ValueError(
                    f"Multiple classes registered with {cls.__name__}. "
                    "Must specify a name."
                )
            name = next(iter(cls.registry.keys()))

        class_type = cls._registry[cls.__name__].get(name, None)
        if class_type is None:
            raise ValueError(f"Class '{name}' not found in {cls.__name__}'s registry.")

        return class_type(*args, **kwargs)

    @classproperty
    def registry(cls: type[Self]) -> dict[str, Any]:
        """
        Get the registry for this class.

        Returns:
            A dictionary mapping class names to class objects or lazy load paths.
        """
        if cls.__name__ not in cls._registry:
            cls._registry[cls.__name__] = {}
        return cls._registry[cls.__name__]

    @classproperty
    def registered(cls: type[Self]) -> Enum:
        """
        Get an enumeration of the registered classes.

        Returns:
            An enumeration of the registered classes.
        """
        return Enum(cls.__name__, {name: name for name in cls.registry})


def register(class_type):
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
