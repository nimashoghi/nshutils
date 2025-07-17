from __future__ import annotations

from ._base import active_patches as active_patches
from ._base import monkey_unpatch as lovely_monkey_unpatch
from ._base import monkey_unpatch as monkey_unpatch
from ._monkey_patch_all import monkey_patch as lovely_monkey_patch
from ._monkey_patch_all import monkey_patch as monkey_patch
from .config import LovelyConfig as LovelyConfig
from .jax_ import jax_monkey_patch as jax_monkey_patch
from .jax_ import jax_repr as jax_repr
from .numpy_ import numpy_monkey_patch as numpy_monkey_patch
from .numpy_ import numpy_repr as numpy_repr
from .pprint_ import pformat as pformat
from .pprint_ import pprint as pprint
from .torch_ import torch_monkey_patch as torch_monkey_patch
from .torch_ import torch_repr as torch_repr

_ = lovely_monkey_patch
_ = lovely_monkey_unpatch
