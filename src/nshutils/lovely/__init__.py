from __future__ import annotations

from ._monkey_patch_all import monkey_patch as monkey_patch
from .config import LovelyConfig as LovelyConfig
from .jax_ import jax_monkey_patch as jax_monkey_patch
from .jax_ import jax_repr as jax_repr
from .numpy_ import numpy_monkey_patch as numpy_monkey_patch
from .numpy_ import numpy_repr as numpy_repr
from .torch_ import torch_monkey_patch as torch_monkey_patch
from .torch_ import torch_repr as torch_repr

lovely_monkey_patch = monkey_patch
