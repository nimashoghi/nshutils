from __future__ import annotations

from . import actsave as actsave
from . import typecheck as typecheck
from .actsave import ActLoad as ActLoad
from .actsave import ActSave as ActSave
from .collections import apply_to_collection as apply_to_collection
from .display import display as display
from .logging import init_python_logging as init_python_logging
from .logging import lovely as lovely
from .logging import pretty as pretty
from .snoop import snoop as snoop
from .typecheck import tassert as tassert
from .typecheck import typecheck_modules as typecheck_modules
from .typecheck import typecheck_this_module as typecheck_this_module

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # For Python <3.8
    from importlib_metadata import (  # pyright: ignore[reportMissingImports]
        PackageNotFoundError,
        version,
    )

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
