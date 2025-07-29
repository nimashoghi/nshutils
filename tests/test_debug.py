from __future__ import annotations

import os

import nshutils.debug as debug


def test_debug_enabled():
    # Test default state (should match NSHUTILS_DEBUG environment variable)
    expected = bool(int(os.environ.get("NSHUTILS_DEBUG", "0")))
    assert debug.enabled() is expected


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
