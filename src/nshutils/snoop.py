import contextlib
from typing import Any, Protocol, cast

from typing_extensions import TypeVar

T = TypeVar("T", infer_variance=True)


class SnoopConstructor(Protocol):
    def __call__(self, *args, **kwargs) -> contextlib.AbstractContextManager: ...

    def disable(self) -> contextlib.AbstractContextManager: ...


try:
    import warnings
    from contextlib import nullcontext

    import pysnooper
    import pysnooper.utils
    from pkg_resources import DistributionNotFound, get_distribution

    try:
        import torch  # type: ignore
    except ImportError:
        torch = None

    try:
        import numpy  # type: ignore
    except ImportError:
        numpy = None

    FLOATING_POINTS = set()
    for i in ["float", "double", "half", "complex128", "complex32", "complex64"]:
        # older version of PyTorch do not have complex dtypes
        if torch is None or not hasattr(torch, i):
            continue
        FLOATING_POINTS.add(getattr(torch, i))

    try:
        __version__ = get_distribution(__name__).version
    except DistributionNotFound:
        # package is not installed
        pass

    def default_format(x):
        try:
            import lovely_tensors as lt  # type: ignore

            return str(lt.lovely(x))
        except BaseException:
            return str(x.shape)

    def default_numpy_format(x):
        try:
            import lovely_numpy as lo  # type: ignore

            return str(lo.lovely(x))
        except BaseException:
            return str(x.shape)

    class TorchSnooper(pysnooper.tracer.Tracer):
        def __init__(
            self,
            *args,
            tensor_format=default_format,
            numpy_format=default_numpy_format,
            **kwargs,
        ):
            self.orig_custom_repr = (
                kwargs["custom_repr"] if "custom_repr" in kwargs else ()
            )
            custom_repr = (lambda x: True, self.compute_repr)
            kwargs["custom_repr"] = (custom_repr,)
            super(TorchSnooper, self).__init__(*args, **kwargs)
            self.tensor_format = tensor_format
            self.numpy_format = numpy_format

        @staticmethod
        def is_return_types(x):
            return type(x).__module__ == "torch.return_types"

        def return_types_repr(self, x):
            if type(x).__name__ in {
                "max",
                "min",
                "median",
                "mode",
                "sort",
                "topk",
                "kthvalue",
            }:
                return (
                    type(x).__name__
                    + "(values="
                    + self.tensor_format(x.values)
                    + ", indices="
                    + self.tensor_format(x.indices)
                    + ")"
                )
            if type(x).__name__ == "svd":
                return (
                    "svd(U="
                    + self.tensor_format(x.U)
                    + ", S="
                    + self.tensor_format(x.S)
                    + ", V="
                    + self.tensor_format(x.V)
                    + ")"
                )
            if type(x).__name__ == "slogdet":
                return (
                    "slogdet(sign="
                    + self.tensor_format(x.sign)
                    + ", logabsdet="
                    + self.tensor_format(x.logabsdet)
                    + ")"
                )
            if type(x).__name__ == "qr":
                return (
                    "qr(Q="
                    + self.tensor_format(x.Q)
                    + ", R="
                    + self.tensor_format(x.R)
                    + ")"
                )
            if type(x).__name__ == "solve":
                return (
                    "solve(solution="
                    + self.tensor_format(x.solution)
                    + ", LU="
                    + self.tensor_format(x.LU)
                    + ")"
                )
            if type(x).__name__ == "geqrf":
                return (
                    "geqrf(a="
                    + self.tensor_format(x.a)
                    + ", tau="
                    + self.tensor_format(x.tau)
                    + ")"
                )
            if type(x).__name__ in {"symeig", "eig"}:
                return (
                    type(x).__name__
                    + "(eigenvalues="
                    + self.tensor_format(x.eigenvalues)
                    + ", eigenvectors="
                    + self.tensor_format(x.eigenvectors)
                    + ")"
                )
            if type(x).__name__ == "triangular_solve":
                return (
                    "triangular_solve(solution="
                    + self.tensor_format(x.solution)
                    + ", cloned_coefficient="
                    + self.tensor_format(x.cloned_coefficient)
                    + ")"
                )
            if type(x).__name__ == "gels":
                return (
                    "gels(solution="
                    + self.tensor_format(x.solution)
                    + ", QR="
                    + self.tensor_format(x.QR)
                    + ")"
                )
            warnings.warn("Unknown return_types encountered, open a bug report!")

        def compute_repr(self, x):
            orig_repr_func = pysnooper.utils.get_repr_function(x, self.orig_custom_repr)
            if torch is not None and torch.is_tensor(x):
                return self.tensor_format(x)
            if numpy is not None and isinstance(x, numpy.ndarray):
                return self.numpy_format(x)
            if self.is_return_types(x):
                return self.return_types_repr(x)
            if orig_repr_func is not repr:
                return orig_repr_func(x)
            if isinstance(x, (list, tuple)):
                content = ""
                for i in x:
                    if content != "":
                        content += ", "
                    content += self.compute_repr(i)
                if isinstance(x, tuple) and len(x) == 1:
                    content += ","
                if isinstance(x, tuple):
                    return "(" + content + ")"
                return "[" + content + "]"
            if isinstance(x, dict):
                content = ""
                for k, v in x.items():
                    if content != "":
                        content += ", "
                    content += self.compute_repr(k) + ": " + self.compute_repr(v)
                return "{" + content + "}"
            return repr(x)

    class _Snoop:
        disable = nullcontext
        __call__ = TorchSnooper

    snoop: SnoopConstructor = cast(Any, _Snoop())

except ImportError:
    import warnings
    from contextlib import nullcontext

    from typing_extensions import override

    _has_warned = False

    class _snoop_cls(nullcontext):
        @classmethod
        def disable(cls):
            return nullcontext()

        @override
        def __enter__(self):
            global _has_warned
            if not _has_warned:
                warnings.warn(
                    "snoop is not installed, please install it to enable snoop"
                )
                _has_warned = True

            return super().__enter__()

    snoop: SnoopConstructor = cast(Any, _snoop_cls)
