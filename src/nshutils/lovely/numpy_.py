from __future__ import annotations

import logging

import numpy as np

from ._base import lovely_repr, monkey_patch_contextmanager
from .utils import LovelyStats, array_stats


def _np_ge_2():
    import importlib.metadata

    from packaging.version import Version

    try:
        numpy_version = importlib.metadata.version("numpy")
        return Version(numpy_version) >= Version("2.0")
    except importlib.metadata.PackageNotFoundError:
        return False


def _type_name(array: np.ndarray):
    return (
        "array"
        if type(array) is np.ndarray
        else type(array).__name__.rsplit(".", 1)[-1]
    )


_DT_NAMES = {
    "float16": "f16",
    "float32": "f32",
    "float64": "",  # Default dtype in numpy
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "complex64": "c64",
    "complex128": "c128",
}


def _dtype_str(array: np.ndarray) -> str:
    dtype_base = str(array.dtype).rsplit(".", 1)[-1]
    dtype_base = _DT_NAMES.get(dtype_base, dtype_base)
    return dtype_base


@lovely_repr(dependencies=["numpy"])
def numpy_repr(array: np.ndarray) -> LovelyStats | None:
    # For dtypes like `object` or `str`, we let the fallback repr handle it
    if not np.issubdtype(array.dtype, np.number):
        return None

    return {
        # Basic attributes
        "shape": array.shape,
        "size": array.size,
        "nbytes": array.nbytes,
        "type_name": _type_name(array),
        # Dtype
        "dtype_str": _dtype_str(array),
        "is_complex": np.iscomplexobj(array),
        # Depending of whether the tensor is complex or not, we will call the appropriate stats function
        **array_stats(array),
    }


numpy_repr.set_fallback_repr(np.array_repr)


# If numpy 2.0, use the new API override_repr.
if _np_ge_2():

    @monkey_patch_contextmanager(dependencies=["numpy"])
    def numpy_monkey_patch():
        try:
            np.set_printoptions(override_repr=numpy_repr)
            logging.info(
                f"Numpy monkey patching: using {numpy_repr.__name__} for numpy arrays. "
                f"{np.get_printoptions()=}"
            )
            yield
        finally:
            np.set_printoptions(override_repr=None)
            logging.info(
                f"Numpy unmonkey patching: using {numpy_repr.__name__} for numpy arrays. "
                f"{np.get_printoptions()=}"
            )

else:

    @monkey_patch_contextmanager(dependencies=["numpy"])
    def numpy_monkey_patch():
        try:
            np.set_string_function(numpy_repr, True)  # pyright: ignore[reportAttributeAccessIssue]
            np.set_string_function(numpy_repr, False)  # pyright: ignore[reportAttributeAccessIssue]

            logging.info(
                f"Numpy monkey patching: using {numpy_repr.__name__} for numpy arrays. "
                f"{np.get_printoptions()=}"
            )
            yield
        finally:
            np.set_string_function(None, True)  # pyright: ignore[reportAttributeAccessIssue]
            np.set_string_function(None, False)  # pyright: ignore[reportAttributeAccessIssue]

            logging.info(
                f"Numpy unmonkey patching: using {numpy_repr.__name__} for numpy arrays. "
                f"{np.get_printoptions()=}"
            )
