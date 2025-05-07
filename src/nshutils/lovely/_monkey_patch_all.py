from __future__ import annotations

import contextlib
import importlib.util
import logging
from typing import Literal

from typing_extensions import TypeAliasType, assert_never

from ..util import resource_factory_contextmanager

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


@resource_factory_contextmanager
def monkey_patch(libraries: list[Library] | Literal["auto"] = "auto"):
    if libraries == "auto":
        libraries = _find_deps()

    if not libraries:
        raise ValueError(
            "No libraries found for monkey patching. "
            "Please install numpy, torch, or jax."
        )

    with contextlib.ExitStack() as stack:
        for library in libraries:
            match library:
                case "torch":
                    from .torch_ import torch_monkey_patch

                    stack.enter_context(torch_monkey_patch())
                case "jax":
                    from .jax_ import jax_monkey_patch

                    stack.enter_context(jax_monkey_patch())
                case "numpy":
                    from .numpy_ import numpy_monkey_patch

                    stack.enter_context(numpy_monkey_patch())
                case _:
                    assert_never(library)

        log.info(
            f"Monkey patched libraries: {', '.join(libraries)}. "
            "You can now use the lovely functions with these libraries."
        )
        yield
        log.info(
            f"Unmonkey patched libraries: {', '.join(libraries)}. "
            "You can now use the lovely functions with these libraries."
        )
