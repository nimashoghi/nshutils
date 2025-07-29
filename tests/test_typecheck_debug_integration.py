from __future__ import annotations

import os

import pytest

import nshutils.debug as debug
from nshutils.typecheck import _should_typecheck

ENABLE_ENV_KEY = "NSHUTILS_TYPECHECK"
DISABLE_ENV_KEY = "NSHUTILS_DISABLE_TYPECHECKING"


class TestTypecheckDebugIntegration:
    """Test integration between typecheck and debug modules."""

    def setup_method(self):
        """Save original environment and debug state before each test."""
        self.original_enable = os.environ.get(ENABLE_ENV_KEY)
        self.original_disable = os.environ.get(DISABLE_ENV_KEY)
        self.original_debug = os.environ.get("NSHUTILS_DEBUG")

        # Clear environment variables
        for key in [ENABLE_ENV_KEY, DISABLE_ENV_KEY, "NSHUTILS_DEBUG"]:
            if key in os.environ:
                del os.environ[key]

    def teardown_method(self):
        """Restore original environment and debug state after each test."""
        # Restore original environment
        for key, original in [
            (ENABLE_ENV_KEY, self.original_enable),
            (DISABLE_ENV_KEY, self.original_disable),
            ("NSHUTILS_DEBUG", self.original_debug),
        ]:
            if original is not None:
                os.environ[key] = original
            elif key in os.environ:
                del os.environ[key]

        # Reset debug state
        debug.set(None)

    def test_typecheck_disabled_by_default(self):
        """Test that typecheck is disabled by default when debug is off."""
        debug.set(False)
        assert not _should_typecheck()

    def test_typecheck_enabled_when_debug_enabled(self):
        """Test that typecheck is enabled when debug is enabled."""
        debug.set(True)
        assert _should_typecheck()

    def test_explicit_disable_overrides_debug(self):
        """Test that explicit typecheck disable overrides debug state."""
        debug.set(True)
        os.environ[DISABLE_ENV_KEY] = "1"
        assert not _should_typecheck()

    def test_explicit_enable_works_regardless_of_debug(self):
        """Test that explicit typecheck enable works regardless of debug state."""
        debug.set(False)
        os.environ[ENABLE_ENV_KEY] = "1"
        assert _should_typecheck()

    def test_typecheck_with_debug_context_manager(self):
        """Test that typecheck respects debug context manager."""
        debug.set(False)
        assert not _should_typecheck()

        with debug.override(True):
            assert _should_typecheck()

        assert not _should_typecheck()

    def test_priority_order(self):
        """Test the priority order of typecheck configuration."""
        # Test that both env vars cannot be set simultaneously
        debug.set(True)
        os.environ[ENABLE_ENV_KEY] = "1"
        os.environ[DISABLE_ENV_KEY] = "1"

        with pytest.raises(RuntimeError, match="Cannot set both"):
            _should_typecheck()

        # Clean up for remaining tests
        del os.environ[ENABLE_ENV_KEY]
        del os.environ[DISABLE_ENV_KEY]

        # 1. DISABLE has highest priority (when set alone)
        debug.set(True)
        os.environ[DISABLE_ENV_KEY] = "1"
        assert not _should_typecheck()

        # 2. ENABLE has second priority (when DISABLE not set)
        del os.environ[DISABLE_ENV_KEY]
        debug.set(False)
        os.environ[ENABLE_ENV_KEY] = "1"
        assert _should_typecheck()

        # 3. Debug state has third priority
        del os.environ[ENABLE_ENV_KEY]
        debug.set(True)
        assert _should_typecheck()

        # 4. Default is disabled
        debug.set(False)
        assert not _should_typecheck()

    def test_dynamic_decorator_behavior(self):
        """Test that the @typecheck decorator responds to runtime changes."""
        from nshutils.typecheck import typecheck

        # Start with typecheck disabled
        debug.set(False)

        @typecheck
        def add_ints(x: int, y: int) -> int:
            return x + y

        # Should work with wrong types when disabled
        result = add_ints("hello", " world")  # type: ignore
        assert result == "hello world"

        # Enable debug - should now apply type checking
        debug.set(True)

        # Should work with correct types
        assert add_ints(5, 3) == 8

        # Should raise error with wrong types
        with pytest.raises(Exception):  # Could be TypeCheckError or similar
            add_ints("hello", " world")  # type: ignore

        # Disable again - should work with wrong types
        debug.set(False)
        result = add_ints("goodbye", " world")  # type: ignore
        assert result == "goodbye world"
