from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from typing_extensions import override

from ._base import lovely_patch, lovely_repr
from .utils import LovelyStats, array_stats, patch_to

if TYPE_CHECKING:
    import jax  # pyright: ignore[reportMissingImports]


def _type_name(array: jax.Array):
    type_name = type(array).__name__.rsplit(".", 1)[-1]
    return "array" if type_name == "ArrayImpl" else type_name


_DT_NAMES = {
    "float16": "f16",
    "float32": "f32",
    "float64": "f64",
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "bfloat16": "bf16",
    "complex64": "c64",
    "complex128": "c128",
}


def _dtype_str(array: jax.Array) -> str:
    dtype_base = str(array.dtype).rsplit(".", 1)[-1]
    dtype_base = _DT_NAMES.get(dtype_base, dtype_base)
    return dtype_base


def _device(array: jax.Array) -> str:
    from jaxlib.xla_extension import Device  # pyright: ignore[reportMissingImports]

    if callable(device := array.device):
        device = device()

    device = cast(Device, device)
    if device.platform == "cpu":
        return "cpu"

    return f"{device.platform}:{device.id}"


@lovely_repr(dependencies=["jax"])
def jax_repr(array: jax.Array) -> LovelyStats | None:
    import jax.numpy as jnp  # pyright: ignore[reportMissingImports]

    # For dtypes like `object` or `str`, we let the fallback repr handle it
    if not jnp.issubdtype(array.dtype, jnp.number):
        return None

    return {
        # Basic attributes
        "shape": array.shape,
        "size": array.size,
        "nbytes": array.nbytes,
        "type_name": _type_name(array),
        # Dtype
        "dtype_str": _dtype_str(array),
        "is_complex": jnp.iscomplexobj(array),
        # Device
        "device": _device(array),
        # Depending of whether the tensor is complex or not, we will call the appropriate stats function
        **array_stats(np.asarray(array)),
    }


class jax_monkey_patch(lovely_patch):
    @override
    def dependencies(self) -> list[str]:
        return ["jax"]

    @override
    def patch(self):
        from jax._src import array  # pyright: ignore[reportMissingImports]

        self.prev_repr = array.ArrayImpl.__repr__
        self.prev_str = array.ArrayImpl.__str__
        jax_repr.set_fallback_repr(self.prev_repr)

        patch_to(array.ArrayImpl, "__repr__", jax_repr)
        patch_to(array.ArrayImpl, "__str__", jax_repr)

    @override
    def unpatch(self):
        from jax._src import array  # pyright: ignore[reportMissingImports]

        patch_to(array.ArrayImpl, "__repr__", self.prev_repr)
        patch_to(array.ArrayImpl, "__str__", self.prev_str)
