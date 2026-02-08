# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`nshutils` is an ML-researcher-oriented Python utility library providing runtime typechecking, activation saving/loading, pretty tensor printing, and debugging tools for PyTorch, JAX, and NumPy.

## Build & Development Commands

Package manager is **uv** (not Poetry — recently migrated). The project uses `uv_build` as its build backend.

```bash
# Install all dependencies (including dev + optional extras)
uv sync --all-extras --all-groups

# Run tests (pytest with coverage)
uv run pytest

# Run a single test file or test
uv run pytest tests/test_foo.py
uv run pytest tests/test_foo.py::test_bar

# Lint
uv run ruff check
uv run ruff format --check   # check formatting
uv run ruff format           # auto-format

# Type check
uv run basedpyright

# Full publish pipeline (lint → typecheck → test → build → publish)
bash scripts/publish.sh
```

## Code Architecture

**Source layout:** `src/nshutils/` — uses `lazy_loader` in `__init__.py` so imports stay fast while exposing a clean public API.

### Key Modules

- **`config.py`** — Unified configuration via `ContextVar` (thread-safe) + environment variables (`NSHUTILS_DEBUG`, `NSHUTILS_TYPECHECK`, `NSHUTILS_ACTSAVE`, `NSHUTILS_CONFIG`). Hierarchy: debug=true auto-enables typecheck unless explicitly overridden. Context manager overrides available (`config.debug_override()`, `config.actsave_override()`).

- **`typecheck.py`** — Runtime shape/dtype verification via `jaxtyping` + `beartype`. Provides `@typecheck` decorator and `tassert()` for inline assertions. Includes a `PyTree` dynamic type for subscriptable pytree annotations. Toggled at runtime through the config system.

- **`actsave/`** — Context-aware activation saving/loading. `ActSave.context("name")` prefixes saved tensors. Supports fnmatch-style glob filters to selectively save activations. `ActLoad` for reading saved activations back.

- **`lovely/`** — Monkey-patches tensor/array `__repr__` to show high-signal summaries (shape, min/max/mean) instead of raw data.

- **`snoop.py`** — Wraps `pysnooper` with ML-specific formatters so tracebacks show tensor shapes instead of raw data.

- **`logging.py`** — Logging setup with Rich and Treescope integration for pretty tensor formatting.

- **`debug/`** — Debug assertions and Lightning callbacks.

## Code Conventions

- **`from __future__ import annotations`** is mandatory in every file (enforced by ruff `FA100`/`FA102` rules).
- Use `ruff format` for formatting, `basedpyright` in standard mode for type checking.
- Use modern type syntax: `X | None` not `Optional[X]`, `list[int]` not `List[int]`, `collections.abc` for interfaces.
- Prefer `pathlib.Path` over string paths, `einops` over raw tensor reshape/transpose.
- Use `import nshutils.typecheck as tc` for tensor shape annotations (e.g., `tc.Float[torch.Tensor, "batch seq dim"]`).
- Annotate all public API parameters. Return types are optional (MAY).
- No mutable default arguments — use `default_factory`.
- `logging` module for diagnostics, never `print()` in library code.
- Ruff ignores: `F722`, `F821` (jaxtyping string annotations), `E731`, `E741`.
