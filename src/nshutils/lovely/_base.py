from __future__ import annotations

import functools
import importlib.util
import logging
from collections.abc import Callable, Iterator
from typing import Generic, Optional, cast

from typing_extensions import (
    ParamSpec,
    Protocol,
    TypeAliasType,
    TypeVar,
    override,
    runtime_checkable,
)

from ..util import ContextResource, resource_factory_contextmanager
from .utils import LovelyStats, format_tensor_stats

log = logging.getLogger(__name__)

TArray = TypeVar("TArray", infer_variance=True)
P = ParamSpec("P")

LovelyStatsFn = TypeAliasType(
    "LovelyStatsFn",
    Callable[[TArray], Optional[LovelyStats]],
    type_params=(TArray,),
)


@runtime_checkable
class LovelyReprFn(Protocol[TArray]):
    @property
    def __name__(self) -> str: ...

    def set_fallback_repr(self, repr_fn: Callable[[TArray], str]) -> None: ...
    def __call__(self, value: TArray, /) -> str: ...


def _find_missing_deps(dependencies: list[str]):
    missing_deps: list[str] = []

    for dep in dependencies:
        if importlib.util.find_spec(dep) is not None:
            continue

        missing_deps.append(dep)

    return missing_deps


class lovely_repr(Generic[TArray]):
    @override
    def __init__(
        self,
        dependencies: list[str],
        fallback_repr: Callable[[TArray], str] | None = None,
    ):
        """
        Decorator to create a lovely representation function for an array.

        Args:
            dependencies: List of dependencies to check before running the function.
                If any dependency is not available, the function will not run.
            fallback_repr: A function that takes an array and returns its fallback representation.
        Returns:
            A decorator function that takes a function and returns a lovely representation function.

        Example:
            @lovely_repr(dependencies=["torch"])
            def my_array_stats(array):
                return {...}
        """
        super().__init__()

        if fallback_repr is None:
            fallback_repr = repr

        self._dependencies = dependencies
        self._fallback_repr = fallback_repr

    def set_fallback_repr(self, repr_fn: Callable[[TArray], str]) -> None:
        self._fallback_repr = repr_fn

    def __call__(
        self, array_stats_fn: LovelyStatsFn[TArray], /
    ) -> LovelyReprFn[TArray]:
        @functools.wraps(array_stats_fn)
        def wrapper_fn(array: TArray) -> str:
            if missing_deps := _find_missing_deps(self._dependencies):
                log.warning(
                    f"Missing dependencies: {', '.join(missing_deps)}. "
                    "Skipping lovely representation."
                )
                return self._fallback_repr(array)

            if (stats := array_stats_fn(array)) is None:
                return self._fallback_repr(array)

            return format_tensor_stats(stats)

        wrapper = cast(LovelyReprFn[TArray], wrapper_fn)
        wrapper.set_fallback_repr = self.set_fallback_repr
        return wrapper


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
