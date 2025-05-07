from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lovely._monkey_patch_all import Library


def setup_logging(
    *,
    lovely: bool | list[Library] = False,
    treescope: bool = False,
    treescope_autovisualize_arrays: bool = False,
    rich: bool = False,
    rich_tracebacks: bool = False,
    log_level: int | str | None = logging.INFO,
    log_save_dir: Path | None = None,
):
    if lovely:
        from .lovely._monkey_patch_all import monkey_patch

        monkey_patch("auto" if lovely is True else lovely)

    if treescope:
        try:
            # Check if we're in a Jupyter environment
            from IPython import get_ipython

            if get_ipython() is not None:
                import treescope as _treescope  # type: ignore

                _treescope.basic_interactive_setup(
                    autovisualize_arrays=treescope_autovisualize_arrays
                )
            else:
                logging.info(
                    "Treescope setup is only supported in Jupyter notebooks. Skipping."
                )
        except ImportError:
            logging.info(
                "Failed to import `treescope` or `IPython`. Ignoring `treescope` registration"
            )

    log_handlers: list[logging.Handler] = []
    if log_save_dir:
        log_file = log_save_dir / "logging.log"
        log_file.touch(exist_ok=True)
        log_handlers.append(logging.FileHandler(log_file))

    if rich:
        try:
            from rich.logging import RichHandler  # type: ignore

            log_handlers.append(RichHandler(rich_tracebacks=rich_tracebacks))
        except ImportError:
            logging.info(
                "Failed to import rich. Falling back to default Python logging."
            )

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=log_handlers,
    )
    logging.info(
        "Logging initialized. "
        f"Lovely: {lovely}, Treescope: {treescope}, Rich: {rich}, "
        f"Log level: {log_level}, Log save dir: {log_save_dir}"
    )


init_python_logging = setup_logging
