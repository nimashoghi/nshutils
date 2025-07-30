"""Test ActSave integration with unified config system."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from nshutils import config
from nshutils.actsave import ActSave


def test_actsave_config_integration():
    """Test that ActSave integrates with the unified config system."""
    # Clear any existing state
    ActSave.disable()
    config.set(None, "actsave")

    # Initially disabled
    assert not ActSave.is_enabled
    assert not config.actsave_enabled()

    # Enable via config
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        filters = ["layer*", "attention*"]

        actsave_cfg = config.ActSaveConfig(
            enabled=True, save_dir=str(save_dir), filters=filters
        )
        config.set(actsave_cfg, "actsave")

        # Should auto-enable when checking is_enabled
        assert config.actsave_enabled()
        assert ActSave.is_enabled
        assert config.actsave_save_dir() == save_dir
        assert config.actsave_filters() == filters


def test_actsave_environment_variables():
    """Test ActSave environment variable integration."""
    # Set up environment
    original_env = {}
    for key in ["NSHUTILS_ACTSAVE", "NSHUTILS_ACTSAVE_FILTERS"]:
        original_env[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]

    try:
        # Test NSHUTILS_ACTSAVE=1 (temp directory)
        os.environ["NSHUTILS_ACTSAVE"] = "1"
        os.environ["NSHUTILS_ACTSAVE_FILTERS"] = "encoder.*,decoder.*"

        # Reset config to pick up environment
        config.set(None, "actsave")

        assert config.actsave_enabled()
        assert config.actsave_save_dir() is None  # Should be None for temp directory
        assert config.actsave_filters() == ["encoder.*", "decoder.*"]

        # Test NSHUTILS_ACTSAVE with specific path
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["NSHUTILS_ACTSAVE"] = tmpdir
            config.set(None, "actsave")  # Reset to pick up new env

            assert config.actsave_enabled()
            assert config.actsave_save_dir() == Path(tmpdir)

    finally:
        # Restore environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

        # Clean up
        ActSave.disable()
        config.set(None, "actsave")


def test_actsave_context_manager():
    """Test ActSave context manager with unified config."""
    ActSave.disable()
    config.set(None, "actsave")

    assert not ActSave.is_enabled

    with tempfile.TemporaryDirectory() as tmpdir:
        actsave_cfg = config.ActSaveConfig(
            enabled=True, save_dir=tmpdir, filters=["test*"]
        )

        with config.actsave_override(actsave_cfg):
            assert config.actsave_enabled()
            assert ActSave.is_enabled
            assert config.actsave_save_dir() == Path(tmpdir)
            assert config.actsave_filters() == ["test*"]

        # Should return to disabled state
        assert not config.actsave_enabled()
        # Note: ActSave might still be enabled if it was auto-initialized
