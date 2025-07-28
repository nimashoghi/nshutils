from __future__ import annotations

import nshutils.debug as debug
from nshutils.debug._config import _default_enabled


def test_debug_enabled():
    assert debug.enabled() is _default_enabled()


def test_override_root():
    with debug.override({"enabled": False}):
        assert not debug.enabled()
    with debug.override({"enabled": True}):
        assert debug.enabled()


def test_override_root_nested():
    with debug.override({"enabled": False}):
        with debug.override({"enabled": True}):
            assert debug.enabled()

    with debug.override({"enabled": True}):
        with debug.override({"enabled": False}):
            assert not debug.enabled()
