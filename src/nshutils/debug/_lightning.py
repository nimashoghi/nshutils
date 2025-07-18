from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from ._config import ROOT_PATH, override

try:
    import lightning.pytorch as pl  # pyright: ignore[reportMissingImports]

    class SanityDebugFlagCallback(pl.Callback):
        """Callback to enable debug contexts during sanity check."""

        def __init__(
            self,
            paths: list[str] | None = None,
        ):
            self._paths = paths or [ROOT_PATH, "sanity_check"]
            self._contexts: list[contextlib.AbstractContextManager] = []

        def on_sanity_check_start(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule
        ):
            """Enable debug contexts during sanity check."""
            for path in self._paths:
                context = override({"enabled": True}, path=path)
                self._contexts.append(context)

        def on_sanity_check_end(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule
        ):
            """Restore debug contexts after sanity check."""
            for context in reversed(self._contexts):
                context.__exit__(None, None, None)
            self._contexts.clear()

except ImportError:
    if not TYPE_CHECKING:

        class SanityDebugFlagCallback:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "PyTorch Lightning is not installed. Please install it to use SanityDebugFlagCallback."
                )
