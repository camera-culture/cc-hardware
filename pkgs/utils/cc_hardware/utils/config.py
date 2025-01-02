"""Configuration classes for cc-hardware components."""

from typing import Any

from hydra.utils import get_object
from hydra_config import HydraContainerConfig, config_wrapper
from omegaconf import II  # noqa: F401

from cc_hardware.utils.registry import Registry


@config_wrapper
class CCHardwareConfig(HydraContainerConfig, Registry):
    """Base configuration class for cc-hardware components.

    This defines an additional 'instance' attribute that specifies the class
    to instantiate when creating an instance of the configuration. It is optional in
    that not all configurations will have a corresponding class to instantiate.

    Attributes:
        instance (str, optional): The class to instantiate when creating an instance
            of the configuration. This should be the fully qualified class name.
    """

    instance: str | None = None

    def get_instance(self) -> Any:
        """Get the class instance specified in the configuration.

        Returns:
            The class instance.
        """
        assert (
            self.instance is not None
        ), "No instance class specified in configuration."
        return get_object(f"{type(self).__module__}.{self.instance}")

    def create_instance(self, *args, auto_create: bool = True, **kwargs) -> Any:
        """Create an instance of the class specified in the configuration.

        :attr:`instance` must be set to the fully qualified class name of the class to
        instantiate. If no arguments are provided, the configuration instance will be
        passed as the first argument to the class constructor.

        Args:
            *args: Positional arguments to pass to the class constructor

        Keyword Args:
            auto_create: Whether to automatically create the instance if no arguments
                are provided
            **kwargs: Keyword arguments to pass to the class constructor

        Returns:
            An instance of the specified class.
        """
        assert (
            self.instance is not None
        ), "No instance class specified in configuration."
        if len(args) == 0 and auto_create:
            args = [self]
        return self.get_instance()(*args, **kwargs)
