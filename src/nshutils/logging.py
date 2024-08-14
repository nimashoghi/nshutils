import logging
from pathlib import Path


def init_python_logging(
    *,
    lovely_tensors: bool = False,
    lovely_numpy: bool = False,
    treescope: bool = False,
    treescope_autovisualize_arrays: bool = True,
    rich: bool = False,
    rich_tracebacks: bool = False,
    log_level: int | str | None = logging.INFO,
    log_save_dir: Path | None = None,
):
    if lovely_tensors:
        try:
            import lovely_tensors as _lovely_tensors  # type: ignore

            _lovely_tensors.monkey_patch()
        except ImportError:
            logging.exception(
                "Failed to import `lovely_tensors`. Ignoring pretty PyTorch tensor formatting"
            )

    if lovely_numpy:
        try:
            import lovely_numpy as _lovely_numpy  # type: ignore

            _lovely_numpy.set_config(repr=_lovely_numpy.lovely)
        except ImportError:
            logging.exception(
                "Failed to import `lovely_numpy`. Ignoring pretty numpy array formatting"
            )

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
                logging.exception(
                    "Treescope setup is only supported in Jupyter notebooks. Skipping."
                )
        except ImportError:
            logging.exception(
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
            logging.exception(
                "Failed to import rich. Falling back to default Python logging."
            )

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=log_handlers,
    )


def pretty(
    *,
    lovely_tensors: bool = True,
    lovely_numpy: bool = True,
    treescope: bool = True,
    treescope_autovisualize_arrays: bool = True,
    log_level: int | str | None = logging.INFO,
    log_save_dir: Path | None = None,
    rich_log_handler: bool = False,
    rich_tracebacks: bool = False,
):
    init_python_logging(
        lovely_tensors=lovely_tensors,
        lovely_numpy=lovely_numpy,
        treescope=treescope,
        treescope_autovisualize_arrays=treescope_autovisualize_arrays,
        rich=rich_log_handler,
        log_level=log_level,
        log_save_dir=log_save_dir,
        rich_tracebacks=rich_tracebacks,
    )


def lovely(
    *,
    lovely_tensors: bool = True,
    lovely_numpy: bool = True,
    treescope: bool = True,
    treescope_autovisualize_arrays: bool = True,
    log_level: int | str | None = logging.INFO,
    log_save_dir: Path | None = None,
    rich_log_handler: bool = False,
    rich_tracebacks: bool = False,
):
    pretty(
        lovely_tensors=lovely_tensors,
        lovely_numpy=lovely_numpy,
        treescope=treescope,
        treescope_autovisualize_arrays=treescope_autovisualize_arrays,
        log_level=log_level,
        log_save_dir=log_save_dir,
        rich_log_handler=rich_log_handler,
        rich_tracebacks=rich_tracebacks,
    )
