"""This module provides a wrapper to make asynchronous methods synchronous."""

import asyncio
from typing import Any, Callable


async def _async_wrapper(fn, callback):
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def wrapped_callback(*args, **kwargs):
        result = callback(*args, **kwargs)
        if not future.done():
            loop.call_soon_threadsafe(future.set_result, result)

    fn(wrapped_callback)

    return await future


async def _async_gather_wrapper(fns, callback):
    return await asyncio.gather(*[_async_wrapper(fn, callback) for fn in fns])


def call_async(fn: Callable, callback: Callable) -> Any:
    """Wraps an asynchronous method and returns the result of the callback
    synchronously."""
    return asyncio.run(_async_wrapper(fn, callback))


def call_async_gather(fns: Callable, callback: Callable[[list], Any]) -> Any:
    """Wraps multiple asynchronous methods and returns a list of all the
    callback values."""
    return asyncio.run(_async_gather_wrapper(fns, callback))


def call_async_value(fn: Callable, idx: int = 2) -> Any:
    """Wraps an asynchronous method and returns a specific index in the callback
    list."""
    return call_async(fn, lambda data: data[idx])


__all__ = ["call_async", "call_async_gather", "call_async_value"]
