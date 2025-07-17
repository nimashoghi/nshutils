from __future__ import annotations

import builtins
from typing import Any, Literal

from ._monkey_patch_all import Library, monkey_patch


def pformat(
    obj: Any,
    libraries: list[Library] | Literal["auto"] = "auto",
    *,
    quiet_patch: bool = True,
):
    with monkey_patch(libraries=libraries, quiet=quiet_patch):
        return builtins.repr(obj)


def pprint(
    obj: Any,
    libraries: list[Library] | Literal["auto"] = "auto",
    *,
    quiet_patch: bool = True,
):
    print(pformat(obj, libraries=libraries, quiet_patch=quiet_patch))
