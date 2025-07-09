from __future__ import annotations

import sys

# Before we do anything else, check Python >= 3.10 or raise an error.
if sys.version_info < (3, 10):
    raise RuntimeError(
        "nshutils.typecheck requires Python 3.10 or higher. "
        "Please upgrade your Python version to use this module."
    )

import logging
import os
import typing
from collections.abc import Callable, Sequence
from types import FrameType as _FrameType
from typing import Any, TypeAlias, Union

import wadler_lindig as wl
from beartype import beartype
from jaxtyping import BFloat16 as BFloat16
from jaxtyping import Bool as Bool
from jaxtyping import Complex as Complex
from jaxtyping import Complex64 as Complex64
from jaxtyping import Complex128 as Complex128
from jaxtyping import Float as Float
from jaxtyping import Float16 as Float16
from jaxtyping import Float32 as Float32
from jaxtyping import Float64 as Float64
from jaxtyping import Inexact as Inexact
from jaxtyping import Int as Int
from jaxtyping import Int4 as Int4
from jaxtyping import Int8 as Int8
from jaxtyping import Int16 as Int16
from jaxtyping import Int32 as Int32
from jaxtyping import Int64 as Int64
from jaxtyping import Integer as Integer
from jaxtyping import Key as Key
from jaxtyping import Num as Num
from jaxtyping import Real as Real
from jaxtyping import Shaped as Shaped
from jaxtyping import UInt as UInt
from jaxtyping import UInt4 as UInt4
from jaxtyping import UInt8 as UInt8
from jaxtyping import UInt16 as UInt16
from jaxtyping import UInt32 as UInt32
from jaxtyping import UInt64 as UInt64
from jaxtyping import jaxtyped
from jaxtyping._storage import get_shape_memo, shape_str
from typing_extensions import TypeVar

from ._tree_util import set_pytree_backend as set_pytree_backend

if typing.TYPE_CHECKING:
    # Hack taken directly from `jaxtyping`.
    # Set up to deliberately confuse a static type checker.
    PyTree: TypeAlias = getattr(typing, "foo" + "bar")  # pyright: ignore[reportInvalidTypeForm]
    # What's going on with this madness?
    #
    # At static-type-checking-time, we want `PyTree` to be a type for which both
    # `PyTree` and `PyTree[Foo]` are equivalent to `Any`.
    # (The intention is that `PyTree` be a runtime-only type; there's no real way to
    # do more with static type checkers.)
    #
    # Unfortunately, this isn't possible: `Any` isn't subscriptable. And there's no
    # equivalent way we can fake this using typing annotations. (In some sense the
    # closest thing would be a `Protocol[T]` with no methods, but that's actually the
    # opposite of what we want: that ends up allowing nothing at all.)
    #
    # The good news for us is that static type checkers have an internal escape hatch.
    # If they can't figure out what a type is, then they just give up and allow
    # anything. (I believe this is sometimes called `Unknown`.) Thus, this odd-looking
    # annotation, which static type checkers aren't smart enough to resolve.
else:
    from ._pytree_type_dynamic import PyTree as PyTree

log = logging.getLogger(__name__)

DISABLE_ENV_KEY = "NSHUTILS_DISABLE_TYPECHECKING"


def typecheck_modules(modules: Sequence[str]):
    """
    Typecheck the given modules using `jaxtyping`.

    Args:
        modules: Modules to typecheck.
    """
    # If `DISABLE_ENV_KEY` is set and the environment variable is set, skip
    #   typechecking.
    if DISABLE_ENV_KEY is not None and bool(int(os.environ.get(DISABLE_ENV_KEY, "0"))):
        log.critical(
            f"Type checking is disabled due to the environment variable {DISABLE_ENV_KEY}."
        )
        return

    # Install the jaxtyping import hook for this module.
    from jaxtyping import install_import_hook

    install_import_hook(modules, "beartype.beartype")

    log.critical(f"Type checking the following modules: {modules}")


def _get_frame_package_name_or_none(frame: _FrameType) -> str | None:
    # Taken from `beartype._util.func.utilfuncframe.get_frame_package_name_or_none`.
    assert isinstance(frame, _FrameType), f"{repr(frame)} not stack frame."

    # Fully-qualified name of the parent package of the child module declaring
    # the callable whose code object is that of this stack frame's if that
    # module declares its name *OR* the empty string otherwise (e.g., if that
    # module is either a top-level module or script residing outside any parent
    # package structure).
    frame_package_name = frame.f_globals.get("__package__")

    # Return the name of this parent package.
    return frame_package_name


def typecheck_this_module(additional_modules: Sequence[str] = ()):
    """
    Typecheck the calling module and any additional modules using `jaxtyping`.

    Args:
        additional_modules: Additional modules to typecheck.
    """
    # Get the calling module's name.
    # Here, we can just use beartype's internal implementation behind
    # `beartype_this_package`.

    # Get the calling module's name.
    frame = sys._getframe(1)
    assert frame is not None, "frame is None"
    calling_module_name = _get_frame_package_name_or_none(frame)
    assert calling_module_name is not None, "calling_module_name is None"

    # Typecheck the calling module + any additional modules.
    typecheck_modules((calling_module_name, *additional_modules))


def _make_error_str(input: Any, t: Any) -> str:
    error_components: list[str] = []
    error_components.append("Type checking error:")
    if hasattr(t, "__instancecheck_str__"):
        error_components.append(t.__instancecheck_str__(input))

    error_components.append(wl.pformat(input))
    error_components.append(shape_str(get_shape_memo()))

    return "\n".join(error_components)


def tassert(t: Any, input: object):
    """
    Typecheck the input against the given type.

    Args:
        t: Type to check against.
        input: Input to check.
    """
    __tracebackhide__ = True

    # Ignore typechecking if the environment variable is set.
    if DISABLE_ENV_KEY is not None and bool(int(os.environ.get(DISABLE_ENV_KEY, "0"))):
        return

    assert isinstance(input, t), _make_error_str(input, t)


_TypeOrCallable = TypeVar(
    "_TypeOrCallable",
    bound=Union[type, Callable],
)


def typecheck(fn: _TypeOrCallable, /) -> _TypeOrCallable:
    """
    Decorator to typecheck the function's arguments and return value.
    This decorator uses `jaxtyping` to enforce type checking.
    Args:
        fn: Function to typecheck.
    Returns:
        The typechecked function.
    """

    __tracebackhide__ = True

    # Ignore typechecking if the environment variable is set.
    if DISABLE_ENV_KEY is not None and bool(int(os.environ.get(DISABLE_ENV_KEY, "0"))):
        return fn

    return jaxtyped(typechecker=beartype)(fn)
