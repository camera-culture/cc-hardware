"""
This module provides a metaclass for implementing the Singleton pattern,
ensuring that only one instance of a class is created and shared across
the application.

It includes a combined Singleton and Abstract Base Class (ABC) metaclass
to support both functionalities.

Example:

.. code-block:: python

    from abc import ABC
    from singleton_meta import SingletonABCMeta

    class MySingleton(ABC, metaclass=SingletonABCMeta):
        def __init__(self, value):
            self.value = value

    instance1 = MySingleton(10)
    instance2 = MySingleton.instance()

    assert instance1 is instance2
    assert instance1.value == 10
"""

from abc import ABCMeta
from typing import Self


class SingletonMeta(type):
    """
    A metaclass for implementing the Singleton pattern.

    This ensures that only one instance of the class exists. Additional
    calls to create an instance will return the same existing instance.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Create or retrieve the Singleton instance.

        Args:
            *args: Positional arguments for the class constructor.
            **kwargs: Keyword arguments for the class constructor.

        Returns:
            Self: The single instance of the class.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def instance(cls) -> Self:
        """
        Retrieve the Singleton instance, creating it if necessary.

        Returns:
            Self: The single instance of the class.
        """
        if cls not in cls._instances:
            cls._instances[cls] = cls()
        return cls._instances[cls]


class SingletonABCMeta(ABCMeta, SingletonMeta):
    """
    A metaclass combining Singleton and Abstract Base Class (ABC) functionality.

    This is useful for creating classes that need to enforce the Singleton
    pattern while also being an Abstract Base Class.
    """

    pass
