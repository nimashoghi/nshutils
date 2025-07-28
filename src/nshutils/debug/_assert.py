from __future__ import annotations

from collections.abc import Callable

from ._config import ROOT_PATH, enabled


def assert_(
    condition: bool | Callable[[], bool],
    message: str,
    *,
    path: str = ROOT_PATH,
) -> None:
    """Hierarchical debug assertion.

    Args:
        condition: Condition to check
        message: Error message if assertion fails
        path: Debug path for hierarchical control.

    Examples:
        assert_(tensor.shape[0] > 0, "Batch size must be positive", path="training.data")
    """
    if not enabled(path):
        return

    if callable(condition):
        condition = condition()

    if not condition:
        raise AssertionError(f"Assertion failed at {path}: {message}")
