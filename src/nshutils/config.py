from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Final, TypeAlias, TypedDict, cast

from typing_extensions import Self, TypeVar, assert_never
from typing_extensions import override as override_

log = logging.getLogger(__name__)


class BaseConfig(TypedDict, total=False):
    """Base configuration dictionary that all feature configs should inherit from."""

    pass


class DebugConfig(BaseConfig, total=False):
    """Configuration for debug features."""

    enabled: bool


class TypecheckConfig(BaseConfig, total=False):
    """Configuration for typecheck features."""

    enabled: bool


class ActSaveConfig(BaseConfig, total=False):
    """Configuration for activation saving features."""

    enabled: bool
    save_dir: str
    filters: list[str]


class Config(BaseConfig, total=False):
    """Root configuration containing all feature configurations."""

    debug: DebugConfig
    typecheck: TypecheckConfig
    actsave: ActSaveConfig


_SENTINEL: Final = object()
ROOT_PATH: Final = ""

ValueType: TypeAlias = (
    Config | DebugConfig | TypecheckConfig | ActSaveConfig | bool | None
)


def _parse_env_bool(value: str) -> bool:
    """Parse environment variable as boolean."""
    return value.lower() in ("1", "true", "yes", "on")


def _parse_env_config(env_key: str) -> Config:
    """
    Parse environment configuration from various formats.

    Supports:
    1. JSON: NSHUTILS_CONFIG='{"debug": {"enabled": true}, "typecheck": {"enabled": false}}'
    2. Individual overrides: NSHUTILS_DEBUG=1, NSHUTILS_TYPECHECK=0
    3. Comma-separated: NSHUTILS_CONFIG='debug=true,typecheck=false'
    """
    env_value = os.environ.get(env_key, "").strip()
    config: Config = {}

    if not env_value:
        return config

    # Try JSON first
    if env_value.startswith("{"):
        try:
            parsed = json.loads(env_value)
            if isinstance(parsed, dict):
                return parsed  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass

    # Try comma-separated format
    if "=" in env_value:
        for pair in env_value.split(","):
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key == "debug":
                config.setdefault("debug", {})["enabled"] = _parse_env_bool(value)
            elif key == "typecheck":
                config.setdefault("typecheck", {})["enabled"] = _parse_env_bool(value)
            elif key == "actsave":
                config.setdefault("actsave", {})["enabled"] = _parse_env_bool(value)

    return config


TNew = TypeVar("TNew", infer_variance=True, default=str)
TOld = TypeVar("TOld", infer_variance=True, default=str)


def _getenv_deprecated(
    new_key: str,
    deprecated_key: str,
    transform_fn: Callable[[str], TNew] = lambda x: cast(TNew, x),
    deprecated_transform_fn: Callable[[str], TOld] = lambda x: cast(TOld, x),
    error_on_both: bool = True,
):
    """Get environment variable with deprecation warning."""
    value = os.environ.get(new_key, None)
    deprecated_value = os.environ.get(deprecated_key, None)

    if error_on_both and (value is not None and deprecated_value is not None):
        raise RuntimeError(
            f"Cannot set both {new_key} and {deprecated_key} environment variables at the same time. "
            f"{deprecated_key} is deprecated, use {new_key} instead."
        )

    if value is not None:
        value = transform_fn(cast(str, value))
        return value

    if deprecated_value is not None:
        log.warning(
            f"Environment variable '{deprecated_key}' is deprecated, use '{new_key}' instead."
        )
        value = deprecated_transform_fn(cast(str, deprecated_value))
        return value

    return None


def _default_config() -> Config:
    """Get default configuration from environment variables."""
    config: Config = {}

    # Start with main config
    main_config = _parse_env_config("NSHUTILS_CONFIG")
    config.update(main_config)

    # Individual environment variable overrides
    debug_env = os.environ.get("NSHUTILS_DEBUG")
    if debug_env is not None:
        config.setdefault("debug", {})["enabled"] = _parse_env_bool(debug_env)

    if (
        typecheck := _getenv_deprecated(
            "NSHUTILS_TYPECHECK",
            "NSHUTILS_DISABLE_TYPECHECKING",
            transform_fn=lambda x: _parse_env_bool(x),
            deprecated_transform_fn=lambda x: not _parse_env_bool(x),
        )
    ) is not None:
        config.setdefault("typecheck", {})["enabled"] = typecheck

    # ActSave environment variables
    if (
        actsave_env := _getenv_deprecated(
            "NSHUTILS_ACTSAVE",
            "ACTSAVE",
            transform_fn=lambda x: x,
            deprecated_transform_fn=lambda x: x,
        )
    ) is not None:
        actsave_config = config.setdefault("actsave", {})
        if actsave_env.lower() in ("1", "true"):
            actsave_config["enabled"] = True
            # save_dir will use default (temp directory)
        else:
            actsave_config["enabled"] = True
            actsave_config["save_dir"] = actsave_env

    # ActSave filters

    if (
        actsave_filters_env := _getenv_deprecated(
            "NSHUTILS_ACTSAVE_FILTERS",
            "ACTSAVE_FILTERS",
            transform_fn=lambda x: x,
            deprecated_transform_fn=lambda x: x,
        )
    ) is not None:
        actsave_config = config.setdefault("actsave", {})
        # Parse comma-separated filters, stripping whitespace
        filters = [f.strip() for f in actsave_filters_env.split(",") if f.strip()]
        if filters:
            actsave_config["filters"] = filters

    return config


def _config_for_value(value: ValueType, feature_path: str) -> BaseConfig | object:
    """Convert a value to either a config dict or _SENTINEL."""
    match value:
        case None:
            return _SENTINEL
        case bool():
            # For feature paths like "debug" or "typecheck", create appropriate config
            if feature_path == "debug":
                return DebugConfig(enabled=value)
            elif feature_path == "typecheck":
                return TypecheckConfig(enabled=value)
            elif feature_path == "actsave":
                return ActSaveConfig(enabled=value)
            else:
                # For nested paths like "debug.something", just set enabled
                return {"enabled": value}
        case Mapping():
            return value
        case _:
            assert_never(value)


class _ConfigNode:
    """
    A hierarchical configuration node with inheritance.

    Each node represents a path in the configuration tree (e.g., "debug", "typecheck", "debug.assertions").
    Nodes inherit from their parents unless explicitly overridden.
    """

    __slots__ = ("_path", "_parent", "_var", "_feature")
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

    def _init(self, path: str, parent: "_ConfigNode | None") -> None:
        self._path = path
        self._parent = parent
        self._feature = path.split(".")[0] if path else ""

        # A ContextVar holding either config dict or _SENTINEL for inheritance
        default: BaseConfig | object
        if parent is None:
            # Root starts with _SENTINEL - will call _get_root_default() when accessed
            default = _SENTINEL
        else:
            default = _SENTINEL

        self._var: ContextVar[BaseConfig | object] = ContextVar(
            f"config:{path or '<root>'}", default=default
        )

    def _get_root_default(self) -> BaseConfig:
        """Get the default configuration for the root node (re-reads environment)."""
        return _default_config()

    def _config(self) -> BaseConfig:
        """Get the effective configuration for this node."""
        val = self._var.get()
        if val is _SENTINEL:
            if self._parent is None:
                # Root fallback - always re-read environment
                return self._get_root_default()

            # Inherit from parent
            parent_config = self._parent._config()

            # Navigate to our portion of parent config
            path_parts = self._path.split(".")
            current_config: Any = parent_config

            for part in path_parts:
                if isinstance(current_config, dict) and part in current_config:
                    current_config = current_config[part]
                else:
                    # Path doesn't exist in parent, return empty config
                    return {}

            return (
                cast(BaseConfig, current_config)
                if isinstance(current_config, dict)
                else {}
            )

        return val  # type: ignore[return-value]

    @property
    def config(self) -> BaseConfig:
        """Return the effective configuration for this node."""
        return self._config()

    def __bool__(self) -> bool:
        """Return effective enabled flag."""
        config = self.config
        return config.get("enabled", False) if isinstance(config, dict) else False

    def enabled(self) -> bool:
        """Return effective enabled flag."""
        return bool(self)

    def set(self, value: ValueType) -> None:
        """Set value. None means inherit parent again."""
        self._var.set(_config_for_value(value, self._feature))

    @contextmanager
    def override(self, value: ValueType):
        """Temporarily override value within current context."""
        token = self._var.set(_config_for_value(value, self._feature))
        try:
            yield
        finally:
            self._var.reset(token)

    @override_
    def __repr__(self) -> str:
        return f"<ConfigNode {self._path!r} enabled={bool(self)}>"


# Convenience layer
def node(path: str = ROOT_PATH) -> _ConfigNode:
    """Return the (singleton) node object for path."""
    return _ConfigNode.node(path)


def config(path: str = ROOT_PATH) -> BaseConfig:
    """Return the effective configuration for path."""
    return node(path).config


def enabled(path: str = ROOT_PATH) -> bool:
    """Check if feature at path is enabled."""
    return bool(node(path))


@contextmanager
def override(value: ValueType, path: str = ROOT_PATH):
    """Temporarily override path within the current async context."""
    with node(path).override(value):
        yield


def set(value: ValueType, path: str = ROOT_PATH) -> None:
    """Set value for path."""
    node(path).set(value)


# Feature-specific convenience functions
def debug_enabled() -> bool:
    """Check if debug is enabled."""
    return enabled("debug")


def typecheck_enabled() -> bool:
    """
    Check if typecheck is enabled.

    Logic:
    1. If typecheck is explicitly configured, use that value
    2. If debug is enabled, enable typecheck as well (hierarchy)
    3. Otherwise, typecheck is disabled
    """
    typecheck_node = node("typecheck")
    typecheck_config = typecheck_node.config

    # If typecheck has an explicit enabled setting, use it
    if "enabled" in typecheck_config:
        return typecheck_config["enabled"]

    # If debug is enabled, enable typecheck by default
    if debug_enabled():
        return True

    # Default: disabled
    return False


@contextmanager
def debug_override(value: bool | DebugConfig | None):
    """Temporarily override debug configuration."""
    with override(value, "debug"):
        yield


@contextmanager
def typecheck_override(value: bool | TypecheckConfig | None):
    """Temporarily override typecheck configuration."""
    with override(value, "typecheck"):
        yield


def actsave_enabled() -> bool:
    """Check if actsave is enabled."""
    return enabled("actsave")


def actsave_config() -> ActSaveConfig:
    """Get the effective actsave configuration."""
    return cast(ActSaveConfig, config("actsave"))


def actsave_save_dir() -> Path | None:
    """Get the actsave save directory, if configured."""
    cfg = actsave_config()
    save_dir_str = cfg.get("save_dir")
    return Path(save_dir_str) if save_dir_str else None


def actsave_filters() -> list[str] | None:
    """Get the actsave filters, if configured."""
    cfg = actsave_config()
    return cfg.get("filters")


@contextmanager
def actsave_override(value: bool | ActSaveConfig | None):
    """Temporarily override actsave configuration."""
    with override(value, "actsave"):
        yield
