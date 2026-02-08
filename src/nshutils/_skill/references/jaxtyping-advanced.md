# jaxtyping Advanced Features

Advanced features of jaxtyping that nshutils also supports via `import nshutils.typecheck as tc`.

## Contents
- Symbolic Dimensions
- Dimension Expressions
- Nested Annotations
- Path-Dependent Axes (PyTree)
- PyTree Structure Matching
- Custom Dtypes
- Import Hook
- Debugging Bindings

## Symbolic Dimensions

Use f-string syntax `{variable}` to reference Python variables in shape strings:

```python
batch_size = 32

@tc.typecheck
def f(x: tc.Float[Tensor, "{batch_size} dim"]):
    ...  # first axis must be exactly 32
```

The variable is evaluated from the local/global scope at call time.

## Dimension Expressions

Return type annotations support basic arithmetic on named dimensions:

```python
@tc.typecheck
def pad(x: tc.Float[Tensor, "seq dim"]) -> tc.Float[Tensor, "seq+2 dim"]:
    return F.pad(x, (0, 0, 1, 1))

@tc.typecheck
def halve(x: tc.Float[Tensor, "seq dim"]) -> tc.Float[Tensor, "seq//2 dim"]:
    return x[::2]
```

Supported operators: `+`, `-`, `*`, `//`. Only valid in **return** annotations (input dims must be bound first).

## Nested Annotations

Compose annotations by subscripting an existing annotation with `Shaped`:

```python
Image = tc.Float[Tensor, "channels height width"]
BatchImage = tc.Shaped[Image, "batch"]
# Equivalent to: tc.Float[Tensor, "batch channels height width"]

@tc.typecheck
def process(images: BatchImage) -> BatchImage:
    ...
```

The outer annotation must use `Shaped` (since the inner already specifies the dtype).

## Path-Dependent Axes (`?`)

For PyTrees where different leaves can have different sizes on a dimension, but corresponding leaves across trees must match:

```python
@tc.typecheck
def tree_add(
    x: tc.PyTree[tc.Float[Tensor, "?batch dim"]],
    y: tc.PyTree[tc.Float[Tensor, "?batch dim"]],
):
    ...
```

`?batch` means: the `batch` size can differ between leaves of `x`, but each leaf in `x` must match the corresponding leaf in `y`.

## PyTree Structure Matching

Bind the tree structure itself to a name:

```python
@tc.typecheck
def tree_map_add(
    x: tc.PyTree[tc.Float[Tensor, "dim"], "T"],
    y: tc.PyTree[tc.Float[Tensor, "dim"], "T"],
) -> tc.PyTree[tc.Float[Tensor, "dim"], "T"]:
    ...  # x, y, and return must have identical nesting structure
```

Structure composition: `"S T"` matches a structure that is the composition of `S` and `T`.
Prefix/suffix matching: `"T ..."` or `"... T"` for partial structure constraints.

## Custom Dtypes

Create custom dtype categories by subclassing `AbstractDtype`:

```python
from jaxtyping import AbstractDtype

class MyCustomDtype(AbstractDtype):
    dtypes = ["float32", "float64"]  # list of dtype strings or regexes

# Usage:
MyCustomDtype[Tensor, "batch dim"]
```

Useful for restricting to a specific subset of dtypes not covered by the built-in hierarchy.

## Import Hook (`install_import_hook`)

Apply `@jaxtyped(typechecker=beartype)` to every annotated function in a package automatically:

```python
from jaxtyping import install_import_hook

# Must be called before importing the target package
install_import_hook("my_package", "beartype.beartype")

import my_package  # all functions in my_package are now typechecked
```

nshutils wraps this via `tc.typecheck_modules(["my_package"])` and `tc.typecheck_this_module()`.

## Debugging Bindings

Inspect current axis name → size mappings at any point during execution:

```python
@tc.typecheck
def f(x: tc.Float[Tensor, "batch dim"]):
    tc.print_bindings()
    # Prints something like: batch=32, dim=128
    ...
```

Useful for debugging shape mismatches — call `tc.print_bindings()` just before the failing assertion.

## Specific Precision Dtypes (Complete List)

**Float**: `Float16`, `Float32`, `Float64`, `BFloat16`, `Float8e4m3fn`, `Float8e5m2`, `Float4e2m1fn`
**Int**: `Int2`, `Int4`, `Int8`, `Int16`, `Int32`, `Int64`
**UInt**: `UInt2`, `UInt4`, `UInt8`, `UInt16`, `UInt32`, `UInt64`
**Complex**: `Complex64`, `Complex128`
**Special**: `Key` (JAX PRNG keys)

Low-precision types (`Int2`, `Int4`, `UInt2`, `UInt4`, `Float8*`) are primarily for JAX.

## `from __future__ import annotations` Caveat

jaxtyping needs annotations to be evaluated at runtime. In nshutils, the `@tc.typecheck` wrapper and `tc.tassert` handle this correctly. However, if you use raw jaxtyping decorators directly (`@jaxtyped`), be aware that `from __future__ import annotations` can sometimes interfere with runtime annotation evaluation. The nshutils wrappers are safe to use with future annotations.
