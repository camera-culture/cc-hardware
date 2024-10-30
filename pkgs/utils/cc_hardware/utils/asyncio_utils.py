import asyncio
from typing import Any


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


def call_async(fn: callable, callback: callable) -> Any:
    """The telemetrix library, when interfacing with the arduino, is asynchronous.
    Return values are passed through a callback, which is annoying. This
    method is a wrapper of any telemetrix method and will return the result of the
    callback synchronously value."""
    return asyncio.run(_async_wrapper(fn, callback))


def call_async_gather(fns: callable, callback: callable[[list], Any]) -> Any:
    """This is a wrapper for the call_async method that returns a list of all the
    callback values."""
    return asyncio.run(_async_gather_wrapper(fns, callback))


def call_async_value(fn: callable, idx: int = 2) -> Any:
    """This is a wrapper for the call_async method that returns a specific index
    in the callback list."""
    return call_async(fn, lambda data: data[idx])
