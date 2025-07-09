from __future__ import annotations

import functools
import importlib
import importlib.util
import logging
from typing import Any, Literal, TypeAlias

from typing_extensions import override

log = logging.getLogger(__name__)


class _Backend:
    def __init__(self, name: str, import_path: str, pytree_def_import: str):
        self.name = name
        self.import_path = import_path
        self.pytree_def_import = pytree_def_import

    def is_available(self) -> bool:
        """Check if the backend is available."""
        spec = importlib.util.find_spec(self.import_path)
        return spec is not None

    @functools.cache
    def _imported_module(self):
        return importlib.import_module(self.import_path)

    def __call__(self, fn_name: str, /, *args, **kwargs):
        module = self._imported_module()
        if (fn := getattr(module, fn_name, None)) is None:
            raise AttributeError(
                f"Module '{self.import_path}' does not implement function '{fn_name}'"
            )
        return fn(*args, **kwargs)

    def pytree_def(self):
        """Return the PyTreeDef class for this backend."""
        # Split the import path to get the class name.
        module_path, cls_name = self.pytree_def_import.rsplit(".", 1)
        module = importlib.import_module(module_path)
        if (pytree_def := getattr(module, cls_name, None)) is None:
            raise AttributeError(
                f"Module '{module_path}' does not implement class '{cls_name}'"
            )
        return pytree_def

    @override
    def __repr__(self) -> str:
        return f"_Backend(name={self.name}, import_path={self.import_path})"

    @override
    def __str__(self) -> str:
        return f"{self.name} ('{self.import_path}')"


_selected_backend: _Backend | None = None

BackendName: TypeAlias = Literal["jax", "torch", "optree"]
DEFAULT_BACKENDS: dict[BackendName, _Backend] = {
    "jax": _Backend("jax", "jax.tree_util", "jax.tree_util.PyTreeDef"),
    "torch": _Backend("torch", "torch.utils._pytree", "torch.utils._pytree.TreeSpec"),
    "optree": _Backend("optree", "optree", "optree.PyTreeSpec"),
}


def set_pytree_backend(name: BackendName):
    """Set the backend for pytree operations."""
    # If not valid backend is provided, raise an error.
    if (backend := DEFAULT_BACKENDS.get(name)) is None:
        raise ValueError(
            f"Invalid backend name '{name}'. Valid options are: {', '.join(DEFAULT_BACKENDS.keys())}."
        )

    global _selected_backend
    if not backend.is_available():
        raise ImportError(f"Backend {backend} is not available.")

    _selected_backend = backend
    log.info(f"Pytree backend set to '{backend}'.")


def _resolve_backend() -> _Backend:
    global _selected_backend
    # If a backend has been explicitly set, use it.
    if _selected_backend is not None:
        return _selected_backend

    # Otherwise, try to find the first available backend from the defaults,
    # however, we should strictly emit a warning if we do so.
    if (
        first_backend := next(
            (b for b in DEFAULT_BACKENDS.values() if b.is_available()), None
        )
    ) is not None:
        log.warning(
            f"Using backend '{first_backend.import_path}' from defaults as no explicit backend is set."
        )
        return first_backend

    raise RuntimeError("No available backend found.")


def _call(fn_name: str, *args: Any, **kwargs: Any) -> Any:
    backend = _resolve_backend()
    return backend(fn_name, *args, **kwargs)


def tree_flatten(*args: Any, **kwargs: Any):
    return _call("tree_flatten", *args, **kwargs)


def tree_unflatten(*args: Any, **kwargs: Any):
    return _call("tree_unflatten", *args, **kwargs)


def tree_map(*args: Any, **kwargs: Any):
    return _call("tree_map", *args, **kwargs)


def tree_structure(*args: Any, **kwargs: Any):
    return _call("tree_structure", *args, **kwargs)


def tree_leaves(*args: Any, **kwargs: Any):
    return _call("tree_leaves", *args, **kwargs)


def PyTreeDef_cls():
    """
    Return the PyTreeDef class for the currently selected backend.
    """
    backend = _resolve_backend()
    return backend.pytree_def()
