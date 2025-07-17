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
    def __init__(
        self,
        libraries: list[Library] | Literal["auto"] = "auto",
        quiet: bool = False,
    ):
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
        self._quiet = quiet
        super().__init__(quiet=quiet)

    @override
    def patch(self):
        for library in self.libraries:
            if library == "torch":
                from .torch_ import torch_monkey_patch

                self.stack.enter_context(torch_monkey_patch(quiet=self._quiet))
            elif library == "jax":
                from .jax_ import jax_monkey_patch

                self.stack.enter_context(jax_monkey_patch(quiet=self._quiet))
            elif library == "numpy":
                from .numpy_ import numpy_monkey_patch

                self.stack.enter_context(numpy_monkey_patch(quiet=self._quiet))
            else:
                assert_never(library)

        self._log(
            f"Monkey patched libraries: {', '.join(self.libraries)}. "
            "You can now use the lovely functions with these libraries."
        )

    @override
    def unpatch(self):
        self.stack.close()
        self._log(
            f"Unmonkey patched libraries: {', '.join(self.libraries)}. "
            "You can now use the lovely functions with these libraries."
        )
