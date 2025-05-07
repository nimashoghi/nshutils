from __future__ import annotations

import functools
import importlib.util
import logging
from collections.abc import Callable, Iterator

from typing_extensions import ParamSpec, TypeAliasType, TypeVar

from ..util import ContextResource, resource_factory_contextmanager
from .utils import LovelyStats, format_tensor_stats

log = logging.getLogger(__name__)

TArray = TypeVar("TArray", infer_variance=True)
P = ParamSpec("P")

LovelyStatsFn = TypeAliasType(
    "LovelyStatsFn",
    Callable[[TArray], LovelyStats],
    type_params=(TArray,),
)
LovelyReprFn = TypeAliasType(
    "LovelyReprFn",
    Callable[[TArray], str],
    type_params=(TArray,),
)


def _find_missing_deps(dependencies: list[str]):
    missing_deps: list[str] = []

    for dep in dependencies:
        if importlib.util.find_spec(dep) is not None:
            continue

        missing_deps.append(dep)

    return missing_deps


def lovely_repr(dependencies: list[str]):
    """
    Decorator to create a lovely representation function for an array.

    Args:
        dependencies: List of dependencies to check before running the function.
            If any dependency is not available, the function will not run.

    Returns:
        A decorator function that takes a function and returns a lovely representation function.

    Example:
        @lovely_repr(dependencies=["torch"])
        def my_array_stats(array):
            return {...}
    """

    def decorator_fn(array_stats_fn: LovelyStatsFn[TArray]) -> LovelyReprFn[TArray]:
        """
        Decorator to create a lovely representation function for an array.

        Args:
            array_stats_fn: A function that takes an array and returns its stats.

        Returns:
            A function that takes an array and returns its lovely representation.
        """

        @functools.wraps(array_stats_fn)
        def wrapper(array: TArray) -> str:
            if missing_deps := _find_missing_deps(dependencies):
                log.warning(
                    f"Missing dependencies: {', '.join(missing_deps)}. "
                    "Skipping lovely representation."
                )
                return repr(array)

            stats = array_stats_fn(array)
            return format_tensor_stats(stats)

        return wrapper

    return decorator_fn


LovelyMonkeyPatchInputFn = TypeAliasType(
    "LovelyMonkeyPatchInputFn",
    Callable[P, Iterator[None]],
    type_params=(P,),
)
LovelyMonkeyPatchFn = TypeAliasType(
    "LovelyMonkeyPatchFn",
    Callable[P, ContextResource[None]],
    type_params=(P,),
)


def _nullcontext_generator():
    """A generator that does nothing."""
    yield


def _wrap_monkey_patch_fn(
    monkey_patch_fn: LovelyMonkeyPatchInputFn[P],
    dependencies: list[str],
) -> LovelyMonkeyPatchInputFn[P]:
    @functools.wraps(monkey_patch_fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Iterator[None]:
        if missing_deps := _find_missing_deps(dependencies):
            log.warning(
                f"Missing dependencies: {', '.join(missing_deps)}. "
                "Skipping monkey patch."
            )
            return _nullcontext_generator()

        return monkey_patch_fn(*args, **kwargs)

    return wrapper


def monkey_patch_contextmanager(dependencies: list[str]):
    """
    Decorator to create a monkey patch function for an array.

    Args:
        dependencies: List of dependencies to check before running the function.
            If any dependency is not available, the function will not run.

    Returns:
        A decorator function that takes a function and returns a monkey patch function.

    Example:
        @monkey_patch_contextmanager(dependencies=["torch"])
        def my_array_monkey_patch():
            ...
    """

    def decorator_fn(
        monkey_patch_fn: LovelyMonkeyPatchInputFn[P],
    ) -> LovelyMonkeyPatchFn[P]:
        """
        Decorator to create a monkey patch function for an array.

        Args:
            monkey_patch_fn: A function that applies the monkey patch.

        Returns:
            A function that applies the monkey patch.
        """

        wrapped_fn = _wrap_monkey_patch_fn(monkey_patch_fn, dependencies)
        return resource_factory_contextmanager(wrapped_fn)

    return decorator_fn
