"""This module defines the base classes for sensor settings and provides
utilities for defining and managing sensor settings."""

from dataclasses import Field, field
from enum import Enum
from numbers import Number
from typing import Any, Type

from cc_hardware.utils import Config, config_wrapper


@config_wrapper
class Setting(Config):
    """
    Defines an individual setting attribute for a sensor. Should be overridden
    to define different logic for different setting types.

    Attributes:
        title (str | None): The title of the setting, used in the UI. If None,
            the dictionary key will be used.
        dirty (bool): Whether the setting has been modified since the last
            time it was read.

    Note:
        Ideally, this would be generics, but OmegaConf does not support them:
        `731 <https://github.com/omry/omegaconf/issues/731>`_.
    """

    title: str | None = None
    dirty: bool = True

    value: Any

    @classmethod
    def default_factory(cls, **kwargs) -> Field:
        """
        Factory method to create a new SensorSetting instance in a way that
        is compatible with dataclasses.
        """
        return field(default_factory=lambda: cls(**kwargs))


@config_wrapper
class RangeSetting(Setting):
    """
    Defines a setting that has a range of valid values.

    Attributes:
        value (Number): The current value of the setting.
        min (Number): The minimum valid value.
        max (Number): The maximum valid value.
    """

    min: Number
    max: Number

    def update(self, value: Number) -> None:
        """
        Updates the setting with a new value.

        Args:
            value (Number): The new value to set.
        """
        if value < self.min or value > self.max:
            raise ValueError(f"Value {value} is out of range [{self.min}, {self.max}].")
        self.value = value
        self.dirty = True


@config_wrapper
class OptionSetting(Setting):
    """
    Defines a setting that has a set of valid options.

    Attributes:
        value (Any): The current value of the setting.
        options (list[Any]): The valid options for the setting.

    Note:
        Ideally, this would be generics, but OmegaConf does not support them:
        `731 <https://github.com/omry/omegaconf/issues/731>`_.
    """

    value: Any
    options: list[Any]

    enum: Type[Enum] | None = None

    def update(self, value: Any) -> None:
        """
        Updates the setting with a new value.

        Args:
            value (Any): The new value to set.
        """
        if value not in self.options:
            raise ValueError(f"Value {value} is not a valid option.")
        self.value = value
        self.dirty = True

    @classmethod
    def from_enum(cls, enum: Type[Enum], default: Any | None = None, **kwargs) -> Field:
        """
        Helper to create an OptionSetting instance in a way that is compatible
        with dataclasses.
        """
        if default is None:
            # Get the first value in the enum
            default = enum.__members__[list(enum.__members__)[0]]
        return cls.default_factory(value=default, options=list(enum), **kwargs)


@config_wrapper
class BoolSetting(Setting):
    """
    Defines a setting that has a boolean value.

    Attributes:
        value (bool): The current value of the setting.
    """

    value: bool

    def update(self, value: bool) -> None:
        """
        Updates the setting with a new value.

        Args:
            value (bool): The new value to set.
        """
        if not isinstance(value, bool):
            raise ValueError("Value must be a boolean.")
        self.value = value
        self.dirty = True
