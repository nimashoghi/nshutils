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

    def __init__(self, dir: Path):
        self._dir = dir

    def activation(self, name: str):
        return LoadedActivation(self._dir, name)

    @cached_property
    def activations(self):
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

    @override
    def __repr__(self):
        acts_str = pprint.pformat(
            {
                name: f"<{activation.num_activations} activations>"
                for name, activation in self.activations.items()
            }
        )
        acts_str = acts_str.replace("'<", "<").replace(">'", ">")
        return f"ActLoad({acts_str})"

    def get(self, name: str, /, default: T) -> LoadedActivation | T:
        return self.activations.get(name, default)


ActivationLoader = ActLoad
