from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from typing_extensions import TypedDict

from .config import LovelyConfig


class LovelyStats(TypedDict, total=False):
    """Statistics for tensor representation."""

    # Basic tensor information
    shape: Sequence[int]
    size: int
    type_name: str
    dtype_str: str
    device: str | None
    nbytes: int
    requires_grad: bool
    is_meta: bool

    # Content flags
    all_zeros: bool
    has_nan: bool
    has_pos_inf: bool
    has_neg_inf: bool
    is_complex: bool

    # Numeric statistics
    min: float | None
    max: float | None
    mean: float | None
    std: float | None

    # Complex number statistics
    mag_min: float | None
    mag_max: float | None
    real_min: float | None
    real_max: float | None
    imag_min: float | None
    imag_max: float | None

    # Representation
    values_str: str | None


# Formatting utilities
def sci_mode(f: float) -> bool:
    """Determine if a float should be displayed in scientific notation."""
    config = LovelyConfig.instance()
    return (abs(f) < 10**config.threshold_min) or (abs(f) > 10**config.threshold_max)


def pretty_str(x: Any) -> str:
    """Format a number or array for pretty display.

    Works with scalars, numpy arrays, torch tensors, and jax arrays.
    """
    if isinstance(x, int):
        return f"{x}"
    elif isinstance(x, float):
        if x == 0.0:
            return "0."

        sci = (
            sci_mode(x)
            if LovelyConfig.instance().sci_mode is None
            else LovelyConfig.instance().sci_mode
        )
        fmt = f"{{:.{LovelyConfig.instance().precision}{'e' if sci else 'f'}}}"
        return fmt.format(x)
    elif isinstance(x, complex):
        # Handle complex numbers
        real_part = pretty_str(x.real)
        imag_part = pretty_str(abs(x.imag))
        sign = "+" if x.imag >= 0 else "-"
        return f"{real_part}{sign}{imag_part}j"

    # Handle array-like objects
    try:
        if hasattr(x, "ndim") and x.ndim == 0:
            return pretty_str(x.item())
        elif hasattr(x, "shape") and len(x.shape) > 0:
            slices = [pretty_str(x[i]) for i in range(min(x.shape[0], 10))]
            if x.shape[0] > 10:
                slices.append("...")
            return "[" + ", ".join(slices) + "]"
    except:
        pass

    # Fallback
    return str(x)


def sparse_join(items: list[str | None], sep: str = " ") -> str:
    """Join non-empty strings with a separator."""
    return sep.join([item for item in items if item])


def ansi_color(s: str, col: str, use_color: bool = True) -> str:
    """Add ANSI color to a string if use_color is True."""
    if not use_color:
        return s

    style = defaultdict(str)
    style["grey"] = "\x1b[38;2;127;127;127m"
    style["red"] = "\x1b[31m"
    end_style = "\x1b[0m"

    return style[col] + s + end_style


def bytes_to_human(num_bytes: int) -> str:
    """Convert bytes to a human-readable string (b, Kb, Mb, Gb)."""
    units = ["b", "Kb", "Mb", "Gb"]

    value = num_bytes
    matched_unit: str | None = None
    for unit in units:
        if value < 1024 / 10:
            matched_unit = unit
            break
        value /= 1024.0

    assert matched_unit is not None, "No matching unit found"

    if value % 1 == 0 or value >= 10:
        return f"{round(value)}{matched_unit}"
    else:
        return f"{value:.1f}{matched_unit}"


def in_debugger() -> bool:
    """Returns True if running in a debugger."""
    return getattr(sys, "gettrace", None) is not None and sys.gettrace() is not None


# Common tensor representation
def format_tensor_stats(tensor_stats: LovelyStats, color: bool | None = None) -> str:
    """Format tensor stats into a pretty string representation."""
    conf = LovelyConfig.instance()
    if color is None:
        color = conf.color
    if in_debugger():
        color = False

    # Basic tensor info
    shape_str = str(list(shape)) if (shape := tensor_stats.get("shape")) else None
    type_str = sparse_join([tensor_stats.get("type_name"), shape_str], sep="")

    # Calculate memory usage
    numel = None
    if (size := tensor_stats.get("size")) and (nbytes := tensor_stats.get("nbytes")):
        shape = tensor_stats.get("shape", [])

        if shape and max(shape) != size:
            numel = f"n={size}"
            if conf.show_mem_above <= nbytes:
                numel = sparse_join([numel, f"({bytes_to_human(nbytes)})"])
        elif conf.show_mem_above <= nbytes:
            numel = bytes_to_human(nbytes)

    # Handle empty tensors
    if tensor_stats.get("size", 0) == 0:
        common = ansi_color("empty", "grey", color)
    # Handle all zeros
    elif tensor_stats.get("all_zeros"):
        common = ansi_color("all_zeros", "grey", color)
    # Handle complex tensors
    elif tensor_stats.get("is_complex"):
        complex_info = []

        # For magnitude stats
        if (mag_min := tensor_stats.get("mag_min")) is not None and (
            mag_max := tensor_stats.get("mag_max")
        ) is not None:
            complex_info.append(f"|z|∈[{pretty_str(mag_min)}, {pretty_str(mag_max)}]")

        # For real part stats
        if (real_min := tensor_stats.get("real_min")) is not None and (
            real_max := tensor_stats.get("real_max")
        ) is not None:
            complex_info.append(f"Re∈[{pretty_str(real_min)}, {pretty_str(real_max)}]")

        # For imaginary part stats
        if (imag_min := tensor_stats.get("imag_min")) is not None and (
            imag_max := tensor_stats.get("imag_max")
        ) is not None:
            complex_info.append(f"Im∈[{pretty_str(imag_min)}, {pretty_str(imag_max)}]")

        common = sparse_join(complex_info)
    # Handle normal tensors with stats
    elif (min_val := tensor_stats.get("min")) is not None and (
        max_val := tensor_stats.get("max")
    ) is not None:
        minmax = None
        meanstd = None

        if tensor_stats.get("size", 0) > 2:
            minmax = f"x∈[{pretty_str(min_val)}, {pretty_str(max_val)}]"

        if (
            (mean := tensor_stats.get("mean")) is not None
            and (std := tensor_stats.get("std")) is not None
            and tensor_stats.get("size", 0) >= 2
        ):
            meanstd = f"μ={pretty_str(mean)} σ={pretty_str(std)}"

        common = sparse_join([minmax, meanstd])
    else:
        common = None

    # Handle warnings
    warnings = []
    if tensor_stats.get("has_nan"):
        warnings.append(ansi_color("NaN!", "red", color))
    if tensor_stats.get("has_pos_inf"):
        warnings.append(ansi_color("+Inf!", "red", color))
    if tensor_stats.get("has_neg_inf"):
        warnings.append(ansi_color("-Inf!", "red", color))

    attention = sparse_join(warnings)
    common = sparse_join([common, attention])

    # Other tensor attributes
    dtype = tensor_stats.get("dtype_str", "")
    device = tensor_stats.get("device")
    grad = "grad" if tensor_stats.get("requires_grad") else None

    # Format values for small tensors
    vals = None
    if (
        0 < tensor_stats.get("size", 0) <= 10
        and tensor_stats.get("is_meta", False) is False
    ):
        vals = tensor_stats.get("values_str")

    # Join all parts
    result = sparse_join([type_str, dtype, numel, common, grad, device, vals])

    return result


def real_stats(array: np.ndarray) -> LovelyStats:
    stats: LovelyStats = {}

    # Check for special values
    stats["has_nan"] = bool(np.isnan(array).any())
    stats["has_pos_inf"] = bool(np.isposinf(array).any())
    stats["has_neg_inf"] = bool(np.isneginf(array).any())

    # Only compute min/max/mean/std for good data
    good_data = array[np.isfinite(array)]

    if len(good_data) > 0:
        stats["min"] = float(good_data.min())
        stats["max"] = float(good_data.max())
        stats["all_zeros"] = stats["min"] == 0 and stats["max"] == 0 and array.size > 1

        if len(good_data) > 1:
            stats["mean"] = float(good_data.mean())
            stats["std"] = float(good_data.std())

    # Get string representation of values for small tensors
    if 0 < array.size <= 10:
        stats["values_str"] = pretty_str(array)

    return stats


def complex_stats(array: np.ndarray) -> LovelyStats:
    stats: LovelyStats = {}

    # Calculate magnitude (absolute value)
    magnitude = np.abs(array)

    # Check for special values in real or imaginary parts
    stats["has_nan"] = bool(np.isnan(array.real).any() or np.isnan(array.imag).any())

    # Get statistics for magnitude
    good_mag = magnitude[np.isfinite(magnitude)]
    if len(good_mag) > 0:
        stats["mag_min"] = float(good_mag.min())
        stats["mag_max"] = float(good_mag.max())
        stats["all_zeros"] = (
            stats["mag_min"] == 0 and stats["mag_max"] == 0 and array.size > 1
        )

    # Get statistics for real and imaginary parts
    real_part = array.real
    imag_part = array.imag

    good_real = real_part[np.isfinite(real_part)]
    good_imag = imag_part[np.isfinite(imag_part)]

    if len(good_real := real_part[np.isfinite(real_part)]):
        stats["real_min"] = float(good_real.min())
        stats["real_max"] = float(good_real.max())

    if len(good_imag := imag_part[np.isfinite(imag_part)]):
        stats["imag_min"] = float(good_imag.min())
        stats["imag_max"] = float(good_imag.max())

    # Get string representation of values for small tensors
    if 0 < array.size <= 10:
        stats["values_str"] = pretty_str(array)

    return stats


def array_stats(array: np.ndarray, ignore_empty: bool = True) -> LovelyStats:
    """Compute all statistics for a given array.

    Args:
        array (np.ndarray): The input array.
        ignore_empty (bool): If True, ignore empty arrays.

    Returns:
        LovelyStats: A dictionary containing the computed statistics.
    """
    if ignore_empty and array.size == 0:
        return {}

    if np.iscomplexobj(array):
        return complex_stats(array)
    else:
        return real_stats(array)


def patch_to(
    cls: type,
    name: str,
    func: Callable[[Any], Any],
    as_property: bool = False,
) -> None:
    """Simple patch_to implementation to avoid fastcore dependency."""
    if as_property:
        setattr(cls, name, property(func))
    else:
        setattr(cls, name, func)
