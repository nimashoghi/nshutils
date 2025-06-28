from __future__ import annotations

import contextlib
import importlib.util
import logging
from typing import Literal

from typing_extensions import TypeAliasType, assert_never, override

from ._base import lovely_patch

Library = TypeAliasType("Library", Literal["numpy", "torch", "jax"])

log = logging.getLogger(__name__)


def _find_deps() -> list[Library]:
    """
    Find available libraries for monkey patching.
    """
    deps: list[Library] = []
    if importlib.util.find_spec("torch") is not None:
        deps.append("torch")
    if importlib.util.find_spec("jax") is not None:
        deps.append("jax")
    if importlib.util.find_spec("numpy") is not None:
        deps.append("numpy")
    return deps


class monkey_patch(lovely_patch):
    def __init__(self, libraries: list[Library] | Literal["auto"] = "auto"):
        if libraries == "auto":
            self.libraries = _find_deps()
        else:
            self.libraries = libraries

        if not self.libraries:
            raise ValueError(
                "No libraries found for monkey patching. "
                "Please install numpy, torch, or jax."
            )

        self.stack = contextlib.ExitStack()
        super().__init__()

    @override
    def patch(self):
        for library in self.libraries:
            if library == "torch":
                from .torch_ import torch_monkey_patch

                self.stack.enter_context(torch_monkey_patch())
            elif library == "jax":
                from .jax_ import jax_monkey_patch

                self.stack.enter_context(jax_monkey_patch())
            elif library == "numpy":
                from .numpy_ import numpy_monkey_patch

                self.stack.enter_context(numpy_monkey_patch())
            else:
                assert_never(library)

        log.info(
            f"Monkey patched libraries: {', '.join(self.libraries)}. "
            "You can now use the lovely functions with these libraries."
        )

    @override
    def unpatch(self):
        self.stack.close()
        log.info(
            f"Unmonkey patched libraries: {', '.join(self.libraries)}. "
            "You can now use the lovely functions with these libraries."
        )
