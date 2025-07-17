from __future__ import annotations

import os
from collections.abc import MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final, TypedDict

from typing_extensions import Self
from typing_extensions import override as override_


class DebugConfig(TypedDict, total=False):
    enabled: bool


_SENTINEL: Final = object()
ROOT_PATH: Final = ""


def _config_for_value(value: bool | None) -> DebugConfig | object:
    """Convert a bool | None value to either a DebugConfig or _SENTINEL."""
    return _SENTINEL if value is None else DebugConfig(enabled=value)


def _default_enabled():
    return __debug__


class _DebugNode:
    __slots__ = ("_path", "_parent", "_var")
    _registry: MutableMapping[str, Self] = {}

    @classmethod
    def node(cls, path: str) -> Self:
        """Create or fetch the node for path."""
        if path in cls._registry:
            return cls._registry[path]

        if not path:  # root
            parent = None
        else:
            parent_path, _, _ = path.rpartition(".")
            parent = cls.node(parent_path)

        self = super().__new__(cls)
        self._init(path, parent)
        cls._registry[path] = self
        return self

    def _init(self, path: str, parent: "_DebugNode | None") -> None:
        self._path = path
        self._parent = parent
        # A ContextVar holding either DebugConfig or _SENTINEL for inheritance
        default: DebugConfig | object
        if parent is None:
            # Only the root uses __debug__ as default
            default = DebugConfig(enabled=_default_enabled())
        else:
            default = _SENTINEL

        self._var: ContextVar[DebugConfig | object] = ContextVar(
            f"debug:{path or '<root>'}", default=default
        )

    def _config(self) -> DebugConfig:
        val = self._var.get()
        if val is _SENTINEL:
            return (
                self._parent._config() if self._parent else DebugConfig(enabled=False)
            )
        return val  # type: ignore[return-value]

    def __bool__(self) -> bool:
        """Return effective enabled flag."""
        return self._config().get("enabled", False)

    def enabled(self) -> bool:
        """Return effective enabled flag."""
        return bool(self)

    def set(self, value: bool | None) -> None:
        """Set value. None means inherit parent again."""
        self._var.set(_config_for_value(value))

    @contextmanager
    def override(self, value: bool | None):
        """Temporarily override value within current context."""
        token = self._var.set(_config_for_value(value))
        try:
            yield
        finally:
            self._var.reset(token)

    @override_
    def __repr__(self) -> str:
        return f"<DebugNode {self._path!r} enabled={bool(self)}>"


# Convenience layer
def node(path: str = ROOT_PATH) -> _DebugNode:
    """Return the (singleton) node object for path."""
    return _DebugNode.node(path)


def debug(path: str = ROOT_PATH) -> bool:
    """Cheap functional check: if debug("foo.bar"):"""
    return enabled(path)


def enabled(path: str = ROOT_PATH) -> bool:
    """Cheap functional check: if enabled("foo.bar"):"""
    return bool(node(path))


@contextmanager
def override(path: str, value: bool | None):
    """Temporarily override path within the current async context."""
    with node(path).override(value):
        yield


# Optional environment bootstrap
def _bootstrap_from_env(prefix: str = "NSHUTILS_DEBUG_") -> None:
    """
    Support NSHUTILS_DEBUG= (root) or NSHUTILS_DEBUG_TRAINING_MASKS= style flags.

    "1", "true", "yes", "on" -> True
    "0", "false", "no", "off" -> False
    """
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}
    for key, raw in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].lower().replace("_", ".")  # env uses '_' separator
        if raw.lower() in truthy:
            node(path).set(True)
        elif raw.lower() in falsy:
            node(path).set(False)


_bootstrap_from_env()
