"""Shared utility helpers."""

from __future__ import annotations

import asyncio
import importlib
from functools import wraps
from typing import Any, Callable, TypeVar

from ragmax.core.exceptions import DependencyMissing

T = TypeVar("T")


def require_dependency(package: str, extra: str) -> Any:
    """Import *package* or raise :class:`DependencyMissing`."""
    try:
        return importlib.import_module(package)
    except ImportError:
        raise DependencyMissing(package, extra)


def sync_wrapper(async_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Create a synchronous wrapper around an async function.

    If an event loop is already running, schedules on that loop using
    ``run_coroutine_threadsafe``; otherwise uses ``asyncio.run``.
    """

    @wraps(async_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = async_fn(*args, **kwargs)
        if loop is not None and loop.is_running():
            import concurrent.futures

            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()
        return asyncio.run(coro)

    return wrapper


def batched(iterable: list[T], n: int) -> list[list[T]]:
    """Split *iterable* into batches of size *n*."""
    return [iterable[i : i + n] for i in range(0, len(iterable), n)]
