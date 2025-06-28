from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from ._base import lovely_patch, lovely_repr
from .utils import LovelyStats, array_stats, patch_to

if TYPE_CHECKING:
    import torch  # pyright: ignore[reportMissingImports]


def _type_name(tensor: torch.Tensor):
    import torch  # pyright: ignore[reportMissingImports]

    return (
        "tensor"
        if type(tensor) is torch.Tensor
        else type(tensor).__name__.split(".")[-1]
    )


_DT_NAMES = {
    "float32": "",  # Default dtype
    "float16": "f16",
    "float64": "f64",
    "bfloat16": "bf16",
    "uint8": "u8",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "complex32": "c32",
    "complex64": "c64",
    "complex128": "c128",
}


def _dtype_str(tensor: torch.Tensor) -> str:
    dtype_base = str(tensor.dtype).rsplit(".", 1)[-1]
    dtype_base = _DT_NAMES.get(dtype_base, dtype_base)
    return dtype_base


def _to_np(tensor: torch.Tensor) -> np.ndarray:
    import torch  # pyright: ignore[reportMissingImports]

    # Get tensor data as CPU NumPy array for analysis
    t_cpu = tensor.detach().cpu()

    # Convert bfloat16 to float32 for numpy compatibility
    if tensor.dtype == torch.bfloat16:
        t_cpu = t_cpu.to(torch.float32)

    # Convert to NumPy
    t_np = t_cpu.numpy()

    return t_np


@lovely_repr(dependencies=["torch"])
def torch_repr(tensor: torch.Tensor) -> LovelyStats | None:
    return {
        # Basic attributes
        "shape": tensor.shape,
        "size": tensor.numel(),
        "nbytes": tensor.element_size() * tensor.numel(),
        "type_name": _type_name(tensor),
        # Device
        "device": str(tensor.device) if tensor.device else None,
        "is_meta": device.type == "meta" if (device := tensor.device) else False,
        # Grad
        "requires_grad": tensor.requires_grad,
        # Dtype
        "dtype_str": _dtype_str(tensor),
        "is_complex": tensor.is_complex(),
        # Depending of whether the tensor is complex or not, we will call the appropriate stats function
        **array_stats(_to_np(tensor)),
    }


class torch_monkey_patch(lovely_patch):
    @override
    def dependencies(self) -> list[str]:
        return ["torch"]

    @override
    def patch(self):
        import torch  # pyright: ignore[reportMissingImports]

        self.original_repr = torch.Tensor.__repr__
        self.original_str = torch.Tensor.__str__
        self.original_parameter_repr = torch.nn.Parameter.__repr__
        torch_repr.set_fallback_repr(self.original_repr)

        patch_to(torch.Tensor, "__repr__", torch_repr)
        patch_to(torch.Tensor, "__str__", torch_repr)
        try:
            delattr(torch.nn.Parameter, "__repr__")
        except AttributeError:
            pass

    @override
    def unpatch(self):
        import torch  # pyright: ignore[reportMissingImports]

        patch_to(torch.Tensor, "__repr__", self.original_repr)
        patch_to(torch.Tensor, "__str__", self.original_str)
        patch_to(torch.nn.Parameter, "__repr__", self.original_parameter_repr)
