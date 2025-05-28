from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from typing_extensions import Self


@dataclass
class LovelyConfig:
    """
    This class is used to manage the configuration of the Lovely library.
    """

    precision: int = 3
    """Number of digits after the decimal point."""

    threshold_max: int = 3
    """Absolute values larger than 10^3 use scientific notation."""

    threshold_min: int = -4
    """Absolute values smaller than 10^-4 use scientific notation."""

    sci_mode: bool | None = None
    """Force scientific notation (None=auto)."""

    show_mem_above: int = 1024
    """Show memory size if above this threshold (bytes)."""

    color: bool = True
    """Use ANSI colors in text."""

    indent: int = 2
    """Indent for nested representation."""

    config_instance: ClassVar[Self | None] = None
    """Singleton instance of the LovelyConfig class."""

    @classmethod
    def instance(cls) -> Self:
        """
        Get the singleton instance of the LovelyConfig class.
        If it doesn't exist, create it.
        """
        if cls.config_instance is None:
            cls.config_instance = cls()
        return cls.config_instance
