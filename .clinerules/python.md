# Python Style Guide

**Audience:** Senior & near‑senior Python engineers working in machine learning, data science, and scientific computing (PyTorch, JAX, NumPy, SciPy, pandas, matplotlib, etc.).

**Goal:** High‑signal, enforceable rules that make code _readable, type‑safe, reproducible, and production ready_ without slowing research velocity.

Use the **severity tags** below to prioritize:

| Tag        | Meaning                                                       | Enforcement                |
| ---------- | ------------------------------------------------------------- | -------------------------- |
| **MUST**   | Non‑negotiable; CI failure.                                   | Enforced by tooling/tests. |
| **SHOULD** | Strong preference; dev must justify deviation in code review. | Lint warning.              |
| **MAY**    | Optional/idiomatic guidance.                                  | Use judgment.              |

---

## 0. Quickstart Merge Checklist

- [ ] Code formatted with **ruff format**. **MUST**
- [ ] **ruff check** passes (no unignored errors). **MUST**
- [ ] **basedpyright/pyright** passes (strict). **MUST**
- [ ] Unit + integration tests pass (pytest). **MUST**
- [ ] Public APIs have **parameter** type annotations (**MUST**). Return type annotations **MAY** be added when beneficial.
- [ ] No mutable default args. **MUST**
- [ ] Logging (not print) for diagnostics. **MUST**
- [ ] Resources managed with `with`/context managers. **SHOULD**
- [ ] Notebook code factored into modules if reused. **SHOULD**

---

## 1. Core Principles

1. **Clarity > Cleverness.** Write code you’ll still understand in 6 months. Prefer explicitness when ambiguity risks bugs. **MUST**
2. **Consistency Enables Scale.** Follow the guide so diffs surface logic, not formatting churn. **MUST**
3. **Typed APIs = Living Documentation.** Types are the first line of validation and comprehension. **MUST**
4. **Small, Composable Units.** Functions first; compose over inherit. **SHOULD**
5. **Reproducibility Matters.** Deterministic experiments, logged metadata, versioned data/models. **SHOULD**

---

## 2. Tooling & Configuration

**Formatter:** `ruff format` (configure in `pyproject.toml`). Single canonical formatting pass pre‑commit. **MUST**

**Linting:** `ruff check` + rules we enable (imports, style, complexity, common bugbears). Zero unreviewed ignores. **MUST**

**Type Checking:** `basedpyright` (preferred) or `pyright` in strict mode; config in `pyproject.toml` or `pyrightconfig.json`. Treat errors as build‑blocking unless explicitly suppressed with justification. **MUST**

**Pre‑commit:** Install hooks for format → lint → type check → tests (fast subset). CI re‑runs full suite. **SHOULD**

- Developers MAY tag temporary debug code with the comment `# nocheckin`; a pre‑commit hook can refuse commits containing this marker, serving as a safety net during experimentation. **MAY**

**Python Version:** Project declares minimum supported (≥3.11 strongly recommended). Use `from __future__ import annotations` when needed for forward refs (3.11+ makes this default behavior). **MUST**

**Line Length:** 100 chars soft; tooling will wrap. **SHOULD**

---

## 3. Naming & Imports

Follow PEP 8 unless overridden.

- **Modules/Packages:** `lower_snake_case`. **MUST**
- **Classes / Type Aliases / TypeVars:** `PascalCase` (CapWords). **MUST**
- **Functions / Methods / Variables / Args:** `lower_snake_case`. **MUST**
- **Constants:** `UPPER_SNAKE_CASE`; annotate with `Final` when truly constant. **SHOULD**
- Choose **descriptive names**; avoid ambiguous one‑letter vars except conventional loop counters (`i`, `j`), math temps, or type vars. **MUST**

**Imports:**

1. Group: stdlib, third‑party, local; blank line between groups. **MUST**
2. Use **absolute imports** within project; relative (`.`) only within same package when clarity improves. **SHOULD**
3. Avoid wildcard imports; permitted narrowly in package `__init__.py` where API surface is controlled. **MUST**
4. Keep imports top‑level (module scope) except for optional / heavy deps to avoid import‑time cost or cycles (document when deferring). **SHOULD**

---

## 4. Typing Rules

> TL;DR: **Annotate all public APIs fully.**

### 4.1 Functions & Methods

- Annotate **every parameter** for all public functions/methods (**MUST**). Return type annotations **MAY** be included—add them when the return type is non‑trivial, ambiguous, or clearly improves readability.
- Private helpers (`_foo`) MAY omit return type annotations entirely; include them when they add clarity. **MAY**
- Use keyword‑only parameters for configs with many optional knobs; improves call‑site clarity. **SHOULD**

### 4.2 Modern Syntax

- Use PEP 604 unions: `int | str` not `Union[int, str]`. **MUST**
- Built‑in generics: `list[int]`, `dict[str, float]`. **MUST**
- `X | None` instead of `Optional[X]`. **MUST**
- Use `collections.abc` for collection interfaces (`Iterable`, `Mapping`, etc.) **SHOULD**

### 4.3 Advanced Types

- **Type Aliases** for recurring compound types: `Batch = dict[str, np.ndarray]`. **SHOULD**
- **Protocols** for structural typing (e.g., objects with `.predict`); prefer over inheritance when only an interface matters. **SHOULD**
- **Literal\[...]** for constrained string/int values (e.g., modes). **MAY**
- **Final** for constants; **ClassVar** for class attributes not on instances. **SHOULD**

### 4.4 Avoiding `Any`

- Prefer precise types. If you must use `Any`, comment why and narrow ASAP. **MUST**

### 4.5 Array / Tensor Typing

Use **`nshutils.typecheck`** (alias `tc`) for shape+dtype annotated arrays across NumPy, PyTorch, JAX, TensorFlow, MLX.

**Pattern:** `tc.Float[np.ndarray, "height width 3"]` or symbolic dims (`"*batch seq dim"`).

Guidelines:

- Use semantic dim names (`batch`, `time`, `channels`). **SHOULD**
- Integers for fixed dims (`3`). **SHOULD**
- `*` for one‑or‑more axes, `#` broadcastable, `_` unchecked, `...` any shape (dtype only). **MAY**
- Prefer general dtypes (`Float`, `Int`) unless precision critical. **SHOULD**

---

## 5. Functions, Classes & Data Modeling

### 5.1 Functions First

- Default to free functions or small modules for logic; easier to test/compose. **SHOULD**
- Keep functions focused; aim <50 lines; refactor when branching deeply. **SHOULD**
- Use **early returns** (guard clauses) to reduce nesting. **SHOULD**

### 5.2 Classes When Needed

Use classes to:

- Encapsulate _stateful_ behavior that logically belongs together.
- Represent domain objects / configurations.
- Manage resources (but consider context managers/generators).

Avoid deep inheritance trees; use **composition**. **MUST**
Multiple inheritance only for mixins that are stateless + documented. **SHOULD**

### 5.3 Data Containers

- **`dataclasses`** for lightweight structured data; prefer `frozen=True` when immutable semantics help. **SHOULD**
- **`pydantic`** for validated/serialized configs & API payloads. **SHOULD**
- Use `default_factory=` instead of mutable defaults. **MUST**
- Consider `Enum` for controlled vocabularies. **MAY**

---

## 6. Control Flow & Pythonic Idioms

- Use **walrus (`:=`)** when it _reduces repetition_ and improves readability in `if`, `while`, comprehensions. Avoid dense chains. **MAY**
- Use **`match`** for complex structural dispatch (types, literals, shapes) instead of long `if/elif` ladders. **MAY**
- Prefer comprehension syntax for simple transforms (`[f(x) for x in xs if pred(x)]`). Break out when logic grows. **SHOULD**
- Use `enumerate()` not `range(len(...))` when index+value needed. **MUST**
- Use `zip()` for parallel iteration; consider `strict=True` (3.10+) when lengths must match. **SHOULD**
- Truthiness: `if seq:` not `if len(seq) > 0:`. **SHOULD**
- Check None with `is None` / `is not None` (never `==`). **MUST**

---

## 7. Strings & Formatting

- Use **f‑strings** for interpolation. **MUST**
- Raw strings (`r"..."`) for regex/filepath patterns when needed. **MAY**
- Multi‑line text: triple quotes or textwrap.dedent; keep formatting visible. **MAY**

---

## 8. Comments, Docstrings & Documentation

### 8.1 Docstrings

- Public modules/classes/functions require docstrings. **MUST**
- Use **Google style** (Args:, Returns:, Raises:) or project‑standard; be consistent. **SHOULD**
- Document _behavioral contract_ (what, when, side effects, shape/dtype expectations). **MUST**

### 8.2 Inline Comments

- Explain _why_, not _what_ the code already says. **SHOULD**
- Tag technical debt: `# TODO(name,date?):`, `# FIXME:`, `# NOTE:`. Prefer issue links. **SHOULD**

### 8.3 Module Headers

- Top‑of‑file docstring: purpose, key globals, side effects. **SHOULD**

---

## 9. Error Handling & Logging

**Exceptions**

- Raise the most specific built‑in or custom exception. **MUST**
- Never use bare `except:`; catch specific types. Top‑level catch may use `Exception` but must log & (usually) re‑raise. **MUST**
- Chain causes: `raise FooError(...) from err` to preserve trace. **SHOULD**

**Logging**

- Use `logging` (`logger = logging.getLogger(__name__)`). **MUST**
- No `print()` for diagnostics in library code (ok in throwaway notebooks). **MUST**
- Log structured context (ids, shapes, metrics). Consider JSON log formatter for pipelines. **SHOULD**

---

## 10. Resource Management & Filesystem

- Use `with` for files, locks, db sessions, devices (e.g., `h5py.File`, `torch.cuda.stream`). **MUST**
- Use `contextlib` utilities (`contextmanager`, `ExitStack`) for composite cleanup. **SHOULD**
- Prefer **`pathlib.Path`** over raw strings for paths; join with `/`; convert to str only at boundary APIs. **SHOULD**
- Large data: stream (iterators, generators) instead of loading whole set when feasible. **SHOULD**

---

## 11. Reproducibility (Critical for ML)

- Centralize **random seed** management; expose in config; seed NumPy, PyTorch, Python’s `random`, etc. **MUST**
- Document non‑deterministic ops (e.g., some CUDA kernels). Provide toggles for deterministic mode (`torch.use_deterministic_algorithms`). **SHOULD**
- Version datasets, model code, and hyperparameters in experiment logs. **SHOULD**
- Capture environment (package versions, git commit) with each run. **SHOULD**

---

## 12. Testing Strategy

- Use **pytest**. **MUST**
- Mirror project structure: `tests/` matches `src/` packages. **SHOULD**
- Write unit tests for pure logic; integration/system tests for pipelines & training loops. **SHOULD**
- Use fixtures for reusable data/models; keep fast by default; mark slow/large with pytest markers. **SHOULD**
- Assert tensor shapes/dtypes; surface performance regressions with benchmarks when critical. **SHOULD**
- Test error paths (invalid configs, dimension mismatches). **SHOULD**

---

## 13. Handling Optional / Missing Data

- Encode optional values as `T | None`. **MUST**
- Use **sentinels** only when `None` is a valid semantic value; document sentinel meaning. **SHOULD**
- No mutable default args (`def f(x: list[int] = []):` ❌). Use `None` + init or `default_factory` in dataclasses/Pydantic. **MUST**

---

## 14. Notebooks & Interactive Workflows

Notebooks are for exploration; modules are for reuse.

**In Notebooks:**

- First cell: imports, seeds, config paths. **SHOULD**
- Keep execution order linear; re‑run top‑to‑bottom before commit/share. **SHOULD**
- Refactor reusable logic to `src/` modules; import them. **MUST**
- Clear or reduce cell output before commit; consider **Jupytext** pairing with `.py` scripts for diff‑friendly VCS. **SHOULD**
- Avoid hidden state between cells; prefer functions returning data. **SHOULD**

---

## 15. Performance & Memory Notes (ML‑Focused)

- Vectorize with NumPy / tensor ops; avoid Python loops over large arrays. **SHOULD**
- Prefer **einops** operations (`rearrange`, `repeat`, `reduce`, `pack`, `unpack`) over raw tensor methods (`reshape`, `view`, `transpose`, `tile`, `expand`, `cat`, `stack`, `split`, `chunk`). Improves readability and lets einops pick the most efficient (often zero‑copy) path. **SHOULD**
- Minimize CPU⇄GPU transfers; batch operations. **SHOULD**
- Use in‑place ops (_when safe_) in tight loops to reduce memory churn; document side effects. **MAY**
- Cache expensive pure computations with `functools.lru_cache` / memoization. **MAY**
- Profile before optimizing; `%%timeit`, `cProfile`, PyTorch profiler, JAX profiler. **SHOULD**

---

## 16. Project Layout & Packaging

```
project_root/
├── pyproject.toml          # tooling config: ruff, pyright, build
├── README.md
├── src/
│   └── my_pkg/
│       ├── __init__.py     # define public API (__all__)
│       ├── data.py
│       ├── models.py
│       └── utils/
│           └── ...
└── tests/
    ├── conftest.py
    ├── test_data.py
    └── test_models.py
```

Guidelines:

- Use **src layout**; install in editable mode (`pip install -e .`). **MUST**
- `__init__.py` exposes stable API; avoid heavy logic. **MUST**
- Avoid import‑time side effects (start computations) in modules. **MUST**
- Configuration via typed objects (Pydantic) or structured YAML/JSON parsed at startup. **SHOULD**

---

## 17. Code Review Expectations

Reviewers look for:

- Adherence to **MUST** rules; failing code rejected.
- Clear types & docstrings; do I understand the contract?
- Test coverage of new behavior & error cases.
- Readability: would a teammate debug this quickly?
- Performance flags: obvious vectorization/IO issues?

Leave concise, actionable feedback; link to this guide when requesting changes.

---

## 18. Minimal Examples

### Typed Function

```python
from __future__ import annotations
from typing import Final

MAX_EPOCHS: Final[int] = 100

def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Fraction of correct predictions."""
    if pred.shape != target.shape:
        raise ValueError("Shape mismatch")
    return float((pred == target).mean())
```

### Dataclass Config with Default Factory

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TrainConfig:
    data_dir: Path
    batch_size: int = 32
    aug_prob: float = 0.0
    tags: list[str] = field(default_factory=list)  # no mutable default
```

### Tensor Shape Checking

```python
import nshutils.typecheck as tc
import torch

def attention(
    q: tc.Float[torch.Tensor, "*batch q_len dim"],
    k: tc.Float[torch.Tensor, "*batch k_len dim"],
    v: tc.Float[torch.Tensor, "*batch k_len v_dim"],
) -> tc.Float[torch.Tensor, "*batch q_len v_dim"]:
    ...
```

---

## 19. Diff From Vanilla PEP 8 / Common Guides

| Area                 | This Guide                 | PEP 8 / Common | Rationale                                          |
| -------------------- | -------------------------- | -------------- | -------------------------------------------------- |
| Return type hints    | MAY on non‑trivial returns | Optional       | IDEs infer simple cases; annotate when non‑obvious |
| Line length          | 100                        | 79/88          | Wider terminals, ML stack imports                  |
| Array typing         | tc shape annotations       | N/A            | Catch shape/dtype bugs early                       |
| src/ layout          | Required                   | Optional       | Clean packaging, avoids path hacks                 |
| Strict type checking | Required                   | Variable       | Catch bugs in large codebases                      |

---

## 20. Final Notes

A style guide is a _means_ to velocity, not bureaucracy. When a rule blocks legitimate progress, discuss and evolve the guide. Document approved deviations so we learn as a team.

> **If it’s not in CI, it’s not a rule.** Keep tooling current with the guide.
