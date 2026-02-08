---
name: using-nshutils
description: ML utility library for runtime tensor typechecking, activation saving, and debugging. Use when annotating tensor shapes/dtypes with jaxtyping, using @typecheck or tassert, saving/loading model activations with ActSave, configuring nshutils features, or using snoop/lovely for ML debugging.
---

# nshutils

ML research utilities for PyTorch, JAX, and NumPy. Import convention: `import nshutils.typecheck as tc`.

## Configuration

Hierarchical config via environment variables and `ContextVar` overrides. Debug mode auto-enables typecheck.

```bash
NSHUTILS_DEBUG=1              # enables debug + typecheck
NSHUTILS_TYPECHECK=1          # enable/disable typecheck independently
NSHUTILS_ACTSAVE=1            # enable ActSave (temp dir)
NSHUTILS_ACTSAVE="/path"      # enable ActSave with specific dir
NSHUTILS_ACTSAVE_FILTERS="layer*,attention*"  # comma-separated fnmatch patterns
```

Programmatic:

```python
from nshutils import config

config.set(True, "debug")            # enable debug (also enables typecheck)
config.set(True, "typecheck")        # enable typecheck only
config.set({"enabled": True, "save_dir": "/path", "filters": ["layer*"]}, "actsave")

# Temporary overrides
with config.debug_override(False):
    ...  # debug disabled in this scope

with config.typecheck_override(True):
    ...  # typecheck enabled in this scope
```

## Runtime Typechecking (jaxtyping + beartype)

nshutils wraps jaxtyping to provide runtime shape and dtype verification for tensors. This is the primary feature of the library.

### Quick Start

```python
import nshutils.typecheck as tc
import torch

@tc.typecheck
def attention(
    q: tc.Float[torch.Tensor, "batch q_len dim"],
    k: tc.Float[torch.Tensor, "batch k_len dim"],
    v: tc.Float[torch.Tensor, "batch k_len v_dim"],
) -> tc.Float[torch.Tensor, "batch q_len v_dim"]:
    scores = torch.einsum("bqd,bkd->bqk", q, k)
    weights = scores.softmax(dim=-1)
    return torch.einsum("bqk,bkv->bqv", weights, v)
```

When typechecking is enabled, this verifies at runtime that:
- All inputs are float tensors
- Dimension names match across arguments (e.g., `batch` must be the same size in q, k, and v)
- Return shape is correct

### Annotation Syntax: `Dtype[ArrayType, "shape"]`

Three parts: **dtype category**, **array type**, **shape string**.

```python
tc.Float[torch.Tensor, "batch seq dim"]    # float tensor with 3 named dims
tc.Int[np.ndarray, "height width"]         # integer numpy array
tc.Bool[jax.Array, "batch"]                # boolean JAX array
tc.Shaped[torch.Tensor, "..."]             # any dtype, any shape
```

### Dtype Hierarchy

Use the broadest dtype that satisfies your constraint:

| Category | Matches | Common Use |
|----------|---------|------------|
| `Shaped` | Any dtype | When you only care about shape |
| `Num` | Any numeric (int, float, complex) | Generic numeric ops |
| `Real` | Int or float (not complex) | Most ML tensors |
| `Float` | Any float (f16, bf16, f32, f64) | **Most common for ML** |
| `Int` | Any signed int (i8–i64) | Indices, labels |
| `Integer` | Signed or unsigned int | General integer ops |
| `UInt` | Any unsigned int (u8–u64) | Masks, image pixels |
| `Bool` | Boolean | Masks, conditions |
| `Complex` | Complex types | Signal processing |
| `Inexact` | Float or complex | Differentiable ops |

Specific precisions: `Float32`, `Float64`, `Float16`, `BFloat16`, `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`, `UInt32`, `UInt64`, `Complex64`, `Complex128`.

### Shape Syntax

Shapes are **space-separated** strings (no commas). Each token is one axis.

#### Named Dimensions

```python
def f(x: tc.Float[Tensor, "batch seq dim"],
      y: tc.Float[Tensor, "batch seq dim"]):
    ...  # batch, seq, dim must match between x and y
```

Same name across arguments = same size. This is the core feature for catching shape bugs.

#### Fixed Dimensions

```python
tc.Float[Tensor, "batch 3"]         # last dim is exactly 3
tc.Float[Tensor, "224 224 3"]       # fixed image shape
```

#### Variadic Dimensions (`*`)

Matches **zero or more** axes. Only one variadic per annotation.

```python
tc.Float[Tensor, "*batch dim"]      # any number of batch dims + one dim
tc.Float[Tensor, "*batch seq dim"]  # any leading dims, then seq and dim
```

Named variadics bind across arguments:

```python
def f(x: tc.Float[Tensor, "*batch seq dim"],
      y: tc.Float[Tensor, "*batch dim out"]):
    ...  # *batch must be the same tuple of sizes in x and y
```

#### Broadcasting Dimensions (`#`)

Matches size N **or** size 1 (for broadcasting):

```python
def add(x: tc.Float[Tensor, "#batch dim"],
        y: tc.Float[Tensor, "#batch dim"]):
    ...  # one could be (1, 10), other (32, 10) — both valid
```

#### Anonymous Dimensions (`_`)

Matches any size, no cross-argument binding:

```python
tc.Float[Tensor, "batch _ _"]       # 3D, only batch is checked
```

#### Ellipsis (`...`)

Equivalent to `*_` — matches any number of leading dims without binding:

```python
tc.Float[Tensor, "... dim"]         # any shape ending in dim
tc.Float[Tensor, "..."]             # any shape at all (dtype-only check)
```

#### Scalar (empty string)

```python
tc.Float[Tensor, ""]                # scalar tensor (0-dim)
```

#### Documentation Names (`name=size`)

```python
tc.Float[Tensor, "rows=4 cols=8"]   # name is for docs, checks size 4 and 8
```

### The `@typecheck` Decorator

Wraps a function with runtime shape/dtype checking. Checks are **skipped** when typechecking is disabled (zero overhead in production).

```python
@tc.typecheck
def my_fn(x: tc.Float[torch.Tensor, "batch dim"]) -> tc.Float[torch.Tensor, "batch dim"]:
    return x * 2
```

- Dimension names are scoped to the function call
- Same name = same size across all arguments and return value
- Toggled at runtime via `config.set(True/False, "typecheck")` or `NSHUTILS_TYPECHECK=1`

### Inline Assertions with `tassert`

Check shapes mid-function without decorating:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    tc.tassert(tc.Float[torch.Tensor, "batch seq dim"], x)

    hidden = self.linear(x)
    tc.tassert(tc.Float[torch.Tensor, "batch seq hidden"], hidden)

    return self.output(hidden)
```

`tassert` is a no-op when typechecking is disabled.

### `PyTree` Type

For annotating nested structures (dicts, lists, tuples) of tensors:

```python
def process(data: tc.PyTree[tc.Float[torch.Tensor, "batch dim"]]):
    ...  # every leaf must be a float tensor with shape (batch, dim)
```

At static analysis time, `PyTree` resolves to `Any` so it won't cause pyright errors.

### Framework-Agnostic Annotations

Works with any object that has `.shape` and `.dtype`:

```python
tc.Float[torch.Tensor, "b d"]     # PyTorch
tc.Float[np.ndarray, "b d"]       # NumPy
tc.Float[jax.Array, "b d"]        # JAX
tc.Float[Any, "b d"]              # any framework
```

### Module-Wide Typechecking

Apply typechecking to an entire module without decorating every function:

```python
# At the top of your module:
from nshutils.typecheck import typecheck_this_module
typecheck_this_module()  # installs jaxtyping import hook for this package
```

### Debugging Bindings

```python
tc.print_bindings()  # prints current axis name → size mappings
```

For advanced jaxtyping features (symbolic dims, expressions, nested annotations, custom dtypes, path-dependent axes), see [references/jaxtyping-advanced.md](references/jaxtyping-advanced.md).

## Activation Saving (`ActSave`)

Save and load internal model activations for analysis.

```python
from nshutils import ActSave, ActLoad

# Enable (via config or explicitly)
ActSave.enable(save_dir="path/to/activations", filters=["encoder.*"])

# Save activations by name
def forward(self, x):
    hidden = self.encoder(x)
    ActSave({"encoder.hidden": hidden})
    # or equivalently:
    with ActSave.context("encoder"):
        ActSave(hidden=hidden)  # saved as "encoder.hidden"

# Load later
acts = ActLoad.from_latest_version("path/to/activations")
for tensor in acts["encoder.hidden"]:
    print(tensor.shape)
```

Filtering uses fnmatch patterns (`*`, `?`, `[seq]`). Context prefixes compose: nested `ActSave.context` calls build dot-separated paths.

## Pretty Tensor Printing (`lovely`)

Monkey-patches tensor `__repr__` to show shape, dtype, min/max/mean instead of raw data:

```python
from nshutils.lovely import monkey_patch
monkey_patch()
# Now: print(tensor) → "tensor[32, 128] f32 μ=0.01 σ=1.00 ∈[-2.5, 3.1]"
```

## Enhanced Debugging (`snoop`)

Wraps `pysnooper` with ML-aware formatters (shows tensor shapes in traces):

```python
from nshutils.snoop import snoop

@snoop
def train_step(batch):
    ...  # traces show tensor shapes, not raw data
```

Requires `pip install nshutils[snoop]`.

## Rules

- **Every file MUST start with `from __future__ import annotations`**
- Use `import nshutils.typecheck as tc` (not individual imports from jaxtyping)
- Prefer broad dtype categories (`Float` over `Float32`) unless precision matters
- Use semantic dimension names (`batch`, `seq`, `channels`) not generic (`a`, `b`, `c`)
- **Replace shape comments with `tc.tassert`** — shape comments are unchecked and rot; `tassert` is verified at runtime when typechecking is enabled and is a zero-cost no-op in production:

  ```python
  # BAD — comment is never verified, easily becomes stale
  x = self.linear(x)  # (batch, seq, hidden)

  # GOOD — verified at runtime, documents the shape, zero overhead when disabled
  x = self.linear(x)
  tc.tassert(tc.Float[torch.Tensor, "batch seq hidden"], x)
  ```
