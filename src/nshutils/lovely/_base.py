from __future__ import annotations

import contextlib
import functools
import importlib.util
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, Optional, cast

from typing_extensions import (
    ParamSpec,
    Protocol,
    TypeAliasType,
    TypeVar,
    override,
    runtime_checkable,
)

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
    def __lovely_repr_instance__(self) -> lovely_repr[TArray]: ...

    @__lovely_repr_instance__.setter
    def __lovely_repr_instance__(self, value: lovely_repr[TArray]) -> None: ...

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
        wrapper.__lovely_repr_instance__ = self
        wrapper.set_fallback_repr = self.set_fallback_repr
        return wrapper


class lovely_patch(contextlib.AbstractContextManager["lovely_patch"], ABC):
    def __init__(self):
        self._patched = False
        self.__enter__()

    def dependencies(self) -> list[str]:
        """Subclasses can override this to specify the dependencies of the patch."""
        return []

    @abstractmethod
    def patch(self):
        """Subclasses must implement this."""

    @abstractmethod
    def unpatch(self):
        """Subclasses must implement this."""

    @override
    def __enter__(self):
        if self._patched:
            return self

        if missing_deps := _find_missing_deps(self.dependencies()):
            log.warning(
                f"Missing dependencies: {', '.join(missing_deps)}. "
                "Skipping monkey patch."
            )
            return self

        self.patch()
        self._patched = True
        return self

    @override
    def __exit__(self, *exc_info):
        if not self._patched:
            return

        self.unpatch()
        self._patched = False

    def close(self):
        """Explicitly clean up the resource."""
        self.__exit__(None, None, None)
