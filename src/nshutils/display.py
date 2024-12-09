from __future__ import annotations

import importlib.util
from functools import cache
from typing import Any


@cache
def _in_ipython():
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


@cache
def _treescope_installed():
    return importlib.util.find_spec("treescope") is not None


@cache
def _rich_installed():
    return importlib.util.find_spec("rich") is not None


def display(*args: Any):
    """
    Display the given arguments in the current environment.

    If executed in an IPython environment, the display will be handled
    by treescope if installed, or rich if available. If neither are
    installed, it will fall back to IPython's display function. In a
    non-IPython environment, rich will be used if available, otherwise
    the standard print function will be used.

    Args:
        *args: Any objects to display.
    """
    if _in_ipython():
        if _treescope_installed():
            import treescope

            with treescope.active_autovisualizer.set_scoped(
                treescope.ArrayAutovisualizer()
            ):
                treescope.display(*args)
        elif _rich_installed():
            import rich

            rich.print(*args)
        else:
            from IPython.display import display

            display(*args)
    elif _rich_installed():
        import rich

        rich.print(*args)
    else:
        print(*args)
