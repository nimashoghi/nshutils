from __future__ import annotations

import pytest

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


def test_ensure_success():
    with debug.override({"enabled": True}):

        @debug.ensure(lambda result: result > 0, description="result must be positive")
        def myfunction(x: int) -> int:
            return abs(x)

        assert myfunction(5) == 5


def test_ensure_failure():
    with debug.override({"enabled": True}):

        @debug.ensure(lambda result: result > 0, description="result must be positive")
        def myfunction(x: int) -> int:
            return -abs(x)

        with pytest.raises(debug.ViolationError) as exc_info:
            myfunction(5)

        with pytest.raises(debug.ViolationError) as exc_info:
            myfunction(-5)


def test_require_success():
    with debug.override({"enabled": True}):

        @debug.require(lambda x: x > 0, description="x must be positive")
        def myfunction(x: int) -> int:
            return x

        assert myfunction(5) == 5


def test_require_failure():
    with debug.override({"enabled": True}):

        @debug.require(lambda x: x > 0, description="x must be positive")
        def myfunction(x: int) -> int:
            return x

        with pytest.raises(debug.ViolationError) as exc_info:
            myfunction(-5)


def test_ensure_success_ignored():
    with debug.override({"enabled": False}):

        @debug.ensure(lambda result: result > 0, description="result must be positive")
        def myfunction(x: int) -> int:
            return abs(x)

        assert myfunction(5) == 5


def test_ensure_failure_ignored():
    with debug.override({"enabled": False}):

        @debug.ensure(lambda result: result > 0, description="result must be positive")
        def myfunction(x: int) -> int:
            return -abs(x)

        assert myfunction(5) == -5
        assert myfunction(-5) == -5


def test_require_success_ignored():
    with debug.override({"enabled": False}):

        @debug.require(lambda x: x > 0, description="x must be positive")
        def myfunction(x: int) -> int:
            return x

        assert myfunction(5) == 5


def test_require_failure_ignored():
    with debug.override({"enabled": False}):

        @debug.require(lambda x: x > 0, description="x must be positive")
        def myfunction(x: int) -> int:
            return x

        assert myfunction(-5) == -5
