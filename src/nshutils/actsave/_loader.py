from __future__ import annotations

import pprint
from dataclasses import dataclass, field
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import cast, overload

import numpy as np
from typing_extensions import TypeVar, override

log = getLogger(__name__)

T = TypeVar("T", infer_variance=True)


@dataclass
class LoadedActivation:
    base_dir: Path = field(repr=False)
    name: str
    num_activations: int = field(init=False)
    activation_files: list[Path] = field(init=False, repr=False)

    def __post_init__(self):
        if not self.activation_dir.exists():
            raise ValueError(f"Activation dir {self.activation_dir} does not exist")

        # The number of activations = the * of .npy files in the activation dir
        self.activation_files = list(self.activation_dir.glob("*.npy"))
        # Sort the activation files by the numerical index in the filename
        self.activation_files.sort(key=lambda p: int(p.stem))
        self.num_activations = len(self.activation_files)

    @property
    def activation_dir(self) -> Path:
        return self.base_dir / self.name

    def _load_activation(self, item: int):
        activation_path = self.activation_files[item]
        if not activation_path.exists():
            raise ValueError(f"Activation {activation_path} does not exist")
        return cast(np.ndarray, np.load(activation_path, allow_pickle=True))

    @overload
    def __getitem__(self, item: int) -> np.ndarray: ...

    @overload
    def __getitem__(self, item: slice | list[int]) -> list[np.ndarray]: ...

    def __getitem__(
        self, item: int | slice | list[int]
    ) -> np.ndarray | list[np.ndarray]:
        if isinstance(item, int):
            return self._load_activation(item)
        elif isinstance(item, slice):
            return [
                self._load_activation(i)
                for i in range(*item.indices(self.num_activations))
            ]
        elif isinstance(item, list):
            return [self._load_activation(i) for i in item]
        else:
            raise TypeError(f"Invalid type {type(item)} for item {item}")

    def __iter__(self):
        return iter(self[i] for i in range(self.num_activations))

    def __len__(self):
        return self.num_activations

    def all_activations(self):
        return [self[i] for i in range(self.num_activations)]

    @override
    def __repr__(self):
        return f"<LoadedActivation {self.name} ({self.num_activations} activations)>"


class ActLoad:
    @classmethod
    def all_versions(cls, dir: str | Path):
        dir = Path(dir)

        # If the dir is not an activation base directory, we return None
        if not (dir / ".activationbase").exists():
            return None

        # The contents of `dir` should be directories, each of which is a version.
        return [
            (subdir, int(subdir.name)) for subdir in dir.iterdir() if subdir.is_dir()
        ]

    @classmethod
    def is_valid_activation_base(cls, dir: str | Path):
        return cls.all_versions(dir) is not None

    @classmethod
    def from_latest_version(cls, dir: str | Path):
        # The contents of `dir` should be directories, each of which is a version
        # We need to find the latest version
        if (all_versions := cls.all_versions(dir)) is None:
            raise ValueError(f"{dir} is not an activation base directory")

        path, _ = max(all_versions, key=lambda p: p[1])
        return cls(path)

    def __init__(
        self,
        dir: Path | None = None,
        *,
        _base_activations: dict[str, LoadedActivation] | None = None,
        _prefix_chain: list[str] | None = None,
    ):
        """Initialize ActLoad from a directory or from filtered activations.

        Args:
            dir: Path to the activation directory. Required for root ActLoad instances.
            _base_activations: Pre-filtered activations dict. Used internally for prefix filtering.
            _prefix_chain: Chain of prefixes that have been applied. Used for repr.
        """
        self._dir = dir
        self._base_activations = _base_activations
        self._prefix_chain = _prefix_chain or []

    def activation(self, name: str):
        if self._dir is None:
            raise ValueError(
                "Cannot create activation from filtered ActLoad without base directory"
            )
        # For filtered instances, we need to reconstruct the full name
        full_name = "".join(self._prefix_chain) + name
        return LoadedActivation(self._dir, full_name)

    @cached_property
    def activations(self):
        if self._base_activations is not None:
            return self._base_activations

        if self._dir is None:
            raise ValueError("ActLoad requires either dir or _base_activations")

        dirs = list(self._dir.iterdir())
        # Sort the dirs by the last modified time
        dirs.sort(key=lambda p: p.stat().st_mtime)

        return {p.name: LoadedActivation(self._dir, p.name) for p in dirs}

    def __iter__(self):
        return iter(self.activations.values())

    def __getitem__(self, item: str):
        return self.activations[item]

    def __len__(self):
        return len(self.activations)

    def _ipython_key_completions_(self):
        return list(self.activations.keys())

    @override
    def __repr__(self):
        acts_str = pprint.pformat(
            {
                name: f"<{activation.num_activations} activations>"
                for name, activation in self.activations.items()
            }
        )
        acts_str = acts_str.replace("'<", "<").replace(">'", ">")

        if self._prefix_chain:
            prefix_str = "".join(self._prefix_chain)
            return f"ActLoad(prefix='{prefix_str}', {acts_str})"
        else:
            return f"ActLoad({acts_str})"

    def get(self, name: str, /, default: T) -> LoadedActivation | T:
        return self.activations.get(name, default)

    def filter_by_prefix(self, prefix: str) -> ActLoad:
        """Create a filtered view of activations that match the given prefix.

        Args:
            prefix: The prefix to filter by. Only activations whose names start
                   with this prefix will be included in the filtered view.

        Returns:
            A new ActLoad instance that provides access to matching activations
            with the prefix stripped from their keys. Can be chained multiple times.

        Example:
            >>> loader = ActLoad(some_dir)
            >>> # If loader has keys "my.activation.first", "my.activation.second", "other.key"
            >>> filtered = loader.filter_by_prefix("my.activation.")
            >>> filtered["first"]  # Accesses "my.activation.first"
            >>> filtered["second"]  # Accesses "my.activation.second"
            >>> # Can be chained:
            >>> double_filtered = loader.filter_by_prefix("my.").filter_by_prefix("activation.")
            >>> double_filtered["first"]  # Also accesses "my.activation.first"
        """
        filtered_activations = {}
        for name, activation in self.activations.items():
            if name.startswith(prefix):
                # Strip the prefix from the key
                stripped_name = name[len(prefix) :]
                filtered_activations[stripped_name] = activation

        new_prefix_chain = self._prefix_chain + [prefix]
        return ActLoad(
            _base_activations=filtered_activations,
            _prefix_chain=new_prefix_chain,
            dir=self._dir,
        )


ActivationLoader = ActLoad
