"""Decorators for registering CLI commands and running them."""

from inspect import signature
from typing import Callable

import hydra_zen as zen

from cc_hardware.utils import Registry


def register_cli(func: Callable) -> Callable:
    """Register a CLI command.

    Todo:
        Document this function.
    """

    def wrapper(func: Callable) -> Callable:
        # Inspect the function's signature
        sig = signature(func)

        defaults: dict = {}
        for param in sig.parameters.values():
            # Check if the parameter has a type hint
            if param.annotation is not param.empty:
                # Get the type hint and check for a 'registry' attribute
                type_hint = param.annotation
                if not isinstance(type_hint, type):
                    continue

                if issubclass(type_hint, Registry):
                    defaults[param.name] = "???"

                    # Add each item in the registry to the Zen store
                    for name, target in type_hint.registry.items():
                        zen.store(
                            {"_target_": f"{target}.create"},
                            name=name,
                            group=param.name,
                        )

        # Dynamically create the hydra_defaults using the parameter name
        hydra_defaults = ["_self_"] + [
            {name: default} for name, default in defaults.items()
        ]

        # Add the main store configuration
        parameters = {
            param.name: param.default if param.default is not param.empty else None
            for param in sig.parameters.values()
        }
        zen.store(
            zen.make_config(hydra_defaults=hydra_defaults, **parameters),
            name="main",
        )
        zen.store.add_to_hydra_store()

        return func

    if func is None:
        return wrapper
    return wrapper(func)


def run_cli(func: Callable):
    """Run a CLI command.

    Todo:
        Document this function.
    """
    zen.zen(func, unpack_kwargs=True).hydra_main(
        config_path=None, config_name="main", version_base="1.3"
    )
