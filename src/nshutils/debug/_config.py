from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final, TypeAlias, TypedDict

from typing_extensions import Self, assert_never
from typing_extensions import override as override_


class DebugConfig(TypedDict, total=False):
    enabled: bool


_SENTINEL: Final = object()
ROOT_PATH: Final = ""

ValueType: TypeAlias = DebugConfig | bool | None


def _config_for_value(value: ValueType) -> DebugConfig | object:
    """Convert a bool | None value to either a DebugConfig or _SENTINEL."""
    match value:
        case None:
            return _SENTINEL
        case bool():
            return DebugConfig(enabled=value)
        case Mapping():
            return value
        case _:
            assert_never(value)


def _default_enabled():
    return bool(int(os.environ.get("NSHUTILS_DEBUG", "0")))


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

    @property
    def config(self) -> DebugConfig:
        """Return the effective configuration for this node."""
        return self._config()

    def __bool__(self) -> bool:
        """Return effective enabled flag."""
        return self.config.get("enabled", False)

    def enabled(self) -> bool:
        """Return effective enabled flag."""
        return bool(self)

    def set(self, value: ValueType) -> None:
        """Set value. None means inherit parent again."""
        self._var.set(_config_for_value(value))

    @contextmanager
    def override(self, value: ValueType):
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


def config(path: str = ROOT_PATH):
    """Return the effective configuration for path."""
    return node(path).config


def enabled(path: str = ROOT_PATH) -> bool:
    """Cheap functional check: if enabled("foo.bar"):"""
    return bool(node(path))


@contextmanager
def override(value: ValueType, path: str = ROOT_PATH):
    """Temporarily override path within the current async context."""
    with node(path).override(value):
        yield
