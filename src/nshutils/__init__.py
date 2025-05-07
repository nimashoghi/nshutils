from __future__ import annotations

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)


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
