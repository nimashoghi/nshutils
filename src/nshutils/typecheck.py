import os
from collections.abc import Sequence
from logging import getLogger
from typing import Any

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
from jaxtyping._storage import get_shape_memo, shape_str
from typing_extensions import TypeVar

try:
    import torch  # type: ignore
except ImportError:
    torch = None

try:
    import np  # type: ignore
except ImportError:
    np = None


try:
    import jax  # type: ignore
except ImportError:
    jax = None
log = getLogger(__name__)

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


def typecheck_this_module(additional_modules: Sequence[str] = ()):
    """
    Typecheck the calling module and any additional modules using `jaxtyping`.

    Args:
        additional_modules: Additional modules to typecheck.
    """
    # Get the calling module's name.
    # Here, we can just use beartype's internal implementation behind
    # `beartype_this_package`.
    from beartype._util.func.utilfuncframe import get_frame, get_frame_package_name

    # Get the calling module's name.
    assert get_frame is not None, "get_frame is None"
    frame = get_frame(1)
    assert frame is not None, "frame is None"
    calling_module_name = get_frame_package_name(frame)
    assert calling_module_name is not None, "calling_module_name is None"

    # Typecheck the calling module + any additional modules.
    typecheck_modules((calling_module_name, *additional_modules))


def _make_error_str(input: Any, t: Any) -> str:
    error_components: list[str] = []
    error_components.append("Type checking error:")
    if hasattr(t, "__instancecheck_str__"):
        error_components.append(t.__instancecheck_str__(input))
    if torch is not None and torch.is_tensor(input):
        try:
            from lovely_tensors import lovely  # type: ignore

            error_components.append(repr(lovely(input)))
        except BaseException:
            error_components.append(repr(input.shape))
    elif jax is not None and isinstance(input, jax.Array):
        try:
            from lovely_jax import lovely  # type: ignore

            error_components.append(repr(lovely(input)))
        except BaseException:
            error_components.append(repr(input.shape))
    elif np is not None and isinstance(input, np.ndarray):
        try:
            from lovely_numpy import lovely  # type: ignore

            error_components.append(repr(lovely(input)))
        except BaseException:
            error_components.append(repr(input.shape))
    error_components.append(shape_str(get_shape_memo()))

    return "\n".join(error_components)


T = TypeVar("T", infer_variance=True)


def tassert(t: Any, input: T | tuple[T, ...]):
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

    if isinstance(input, tuple):
        for i in input:
            assert isinstance(i, t), _make_error_str(i, t)
        return
    else:
        assert isinstance(input, t), _make_error_str(input, t)
