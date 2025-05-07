from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterator
from typing import Any, Generic

from typing_extensions import ParamSpec, TypeVar, override

R = TypeVar("R")
P = ParamSpec("P")


class ContextResource(contextlib.AbstractContextManager[R], Generic[R]):
    """A class that provides both direct access to a resource and context management."""

    def __init__(self, resource: R, cleanup_func: Callable[[R], Any]):
        self.resource = resource
        self._cleanup_func = cleanup_func

    @override
    def __enter__(self) -> R:
        """When used as a context manager, return the wrapped resource."""
        return self.resource

    @override
    def __exit__(self, *exc_info) -> None:
        """Clean up the resource when exiting the context."""
        self._cleanup_func(self.resource)

    def close(self) -> None:
        """Explicitly clean up the resource."""
        self._cleanup_func(self.resource)


def resource_factory(
    create_func: Callable[P, R], cleanup_func: Callable[[R], None]
) -> Callable[P, ContextResource[R]]:
    """
    Create a factory function that returns a ContextResource.

    Args:
        create_func: Function that creates the resource
        cleanup_func: Function that cleans up the resource

    Returns:
        A function that returns a ContextResource wrapping the created resource
    """

    @functools.wraps(create_func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> ContextResource[R]:
        resource = create_func(*args, **kwargs)
        return ContextResource(resource, cleanup_func)

    return wrapper


def resource_factory_from_context_fn(
    context_func: Callable[P, contextlib.AbstractContextManager[R]],
) -> Callable[P, ContextResource[R]]:
    """
    Create a factory function that returns a ContextResource.

    Args:
        context_func: Function that creates the resource

    Returns:
        A function that returns a ContextResource wrapping the created resource
    """

    @functools.wraps(context_func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> ContextResource[R]:
        context = context_func(*args, **kwargs)
        resource = context.__enter__()
        return ContextResource(resource, lambda _: context.__exit__(None, None, None))

    return wrapper


def resource_factory_contextmanager(
    context_func: Callable[P, Iterator[R]],
) -> Callable[P, ContextResource[R]]:
    """
    Create a factory function that returns a ContextResource.

    Args:
        context_func: Generator function that creates the resource, yields it, and cleans up the resource when done.

    Returns:
        A function that returns a ContextResource wrapping the created resource
    """
    return resource_factory_from_context_fn(contextlib.contextmanager(context_func))
