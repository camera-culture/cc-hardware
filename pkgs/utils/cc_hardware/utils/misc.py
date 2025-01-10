"""Miscellaneous utility functions/classes."""

from typing import Any

from cc_hardware.utils.logger import get_logger

# ======================


class classproperty:
    def __init__(self, func):
        self.fget = func

    def __get__(self, instance, owner):
        return self.fget(owner)


# ======================


def _locate(path: str) -> Any:
    """
    Locate an object by name or dotted path, importing as necessary.
    This is similar to the pydoc function `locate`, except that it checks for
    the module from the given path from back to front.

    This is lifted from
    `hydra <https://github.com/facebookresearch/hydra/blob/main/hydra/utils.py>`_.
    """
    if path == "":
        raise ImportError("Empty path")
    from importlib import import_module
    from types import ModuleType

    parts = [part for part in path.split(".")]
    for part in parts:
        if not len(part):
            raise ValueError(
                f"Error loading '{path}': invalid dotstring."
                + "\nRelative imports are not supported."
            )
    assert len(parts) > 0
    part0 = parts[0]
    try:
        obj = import_module(part0)
    except Exception as exc_import:
        raise ImportError(
            f"Error loading '{path}':\n{repr(exc_import)}"
            + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        except AttributeError as exc_attr:
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = import_module(mod)
                    continue
                except ModuleNotFoundError as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        f"\nAre you sure that '{part}' is "
                        f"importable from module '{parent_dotpath}'?"
                    ) from exc_import
                except Exception as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                    ) from exc_import
            raise ImportError(
                f"Error loading '{path}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
    return obj


def get_object(path: str, *, verbose: bool = True) -> Any | None:
    """
    Look up an entity based on the dotpath.
    Does not perform any type checks on the entity.

    This is lifted from
    `hydra <https://github.com/facebookresearch/hydra/blob/main/hydra/utils.py>`_.
    """
    try:
        return _locate(path)
    except Exception as e:
        if verbose:
            get_logger().error(f"Error getting object at {path} : {e}")
        raise e
