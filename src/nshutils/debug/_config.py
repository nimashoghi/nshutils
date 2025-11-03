from __future__ import annotations

from contextlib import contextmanager
from typing import Final, Literal, overload

from .. import config as _config
from ..config import DebugConfig

# Re-export for compatibility
ROOT_PATH: Final = _config.ROOT_PATH
ValueType = _config.ValueType


@overload
def enabled(path: str = ROOT_PATH, *, tc: Literal[True] = True) -> Literal[True]: ...


@overload
def enabled(path: str = ROOT_PATH, *, tc: Literal[False]) -> bool: ...


def enabled(path: str = ROOT_PATH, *, tc: bool = True) -> bool:
    """Return True if debug is enabled for the given path.

    The `tc` argument is a hack to signal to the type checker that the debug
    branch is always taken (even if the actual return value of `enabled` is `False`).
    This allows us to do type narrowing inside debug branches (a la `if TYPE_CHECKING: ...`).
    """
    full_path = f"debug.{path}" if path else "debug"
    return _config.enabled(full_path)


def set(value: _config.ValueType, path: str = ROOT_PATH) -> None:
    """Set debug configuration for the given path."""
    full_path = f"debug.{path}" if path else "debug"
    _config.set(value, full_path)


@contextmanager
def override(value: _config.ValueType, path: str = ROOT_PATH):
    """Temporarily override debug configuration for the given path."""
    full_path = f"debug.{path}" if path else "debug"
    with _config.override(value, full_path):
        yield


def config_obj(path: str = ROOT_PATH) -> DebugConfig:
    """Return the debug configuration for the given path."""
    full_path = f"debug.{path}" if path else "debug"
    return _config.config(full_path)  # type: ignore[return-value]


# Alias for backward compatibility
config = config_obj
