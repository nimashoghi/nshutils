#!/usr/bin/env python3
"""
Tests for ActSave filtering functionality.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from nshutils.actsave import ActSave


class TestActSaveFiltering:
    """Test suite for ActSave filtering functionality."""

    def setup_method(self):
        """Clean state before each test."""
        ActSave.disable()

    def teardown_method(self):
        """Clean state after each test."""
        ActSave.disable()

    def test_basic_filtering(self):
        """Test basic filtering with simple patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            # Only save activations matching "layer*"
            filters = ["layer*"]

            with ActSave.enabled(save_dir, filters=filters):
                ActSave(
                    layer1_output=np.array([1, 2, 3]),
                    layer2_hidden=np.array([4, 5, 6]),
                    attention_weights=np.array([7, 8, 9]),  # Should NOT be saved
                    decoder_output=np.array([10, 11, 12]),  # Should NOT be saved
                )

            # Check what was actually saved
            activation_dirs = list((save_dir / "0000").iterdir())
            saved_names = {d.name for d in activation_dirs if d.is_dir()}

            expected = {"layer1_output", "layer2_hidden"}
            assert saved_names == expected

    def test_multiple_filters(self):
        """Test filtering with multiple patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            # Save activations matching either "layer*" or "attention*"
            filters = ["layer*", "attention*"]

            with ActSave.enabled(save_dir, filters=filters):
                ActSave(
                    layer1_output=np.array([1, 2, 3]),  # Should be saved
                    attention_weights=np.array([4, 5, 6]),  # Should be saved
                    decoder_output=np.array([7, 8, 9]),  # Should NOT be saved
                    embedding=np.array([10, 11, 12]),  # Should NOT be saved
                )

            activation_dirs = list((save_dir / "0000").iterdir())
            saved_names = {d.name for d in activation_dirs if d.is_dir()}

            expected = {"layer1_output", "attention_weights"}
            assert saved_names == expected

    def test_no_filters_saves_all(self):
        """Test that no filters means all activations are saved."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            with ActSave.enabled(save_dir):  # No filters
                ActSave(
                    layer1_output=np.array([1, 2, 3]),
                    attention_weights=np.array([4, 5, 6]),
                    decoder_output=np.array([7, 8, 9]),
                )

            activation_dirs = list((save_dir / "0000").iterdir())
            saved_names = {d.name for d in activation_dirs if d.is_dir()}

            expected = {"layer1_output", "attention_weights", "decoder_output"}
            assert saved_names == expected

    def test_contextual_filtering(self):
        """Test filtering with context prefixes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            # Only save activations from encoder context
            filters = ["encoder.*"]

            with ActSave.enabled(save_dir, filters=filters):
                # Decoder context - should NOT be saved
                with ActSave.context("decoder"):
                    ActSave(
                        layer1_output=np.array([1, 2, 3]), attention=np.array([4, 5, 6])
                    )

                # Encoder context - should be saved
                with ActSave.context("encoder"):
                    ActSave(
                        layer1_output=np.array([7, 8, 9]),
                        attention=np.array([10, 11, 12]),
                    )

            activation_dirs = list((save_dir / "0000").iterdir())
            saved_names = {d.name for d in activation_dirs if d.is_dir()}

            expected = {"encoder.layer1_output", "encoder.attention"}
            assert saved_names == expected

    def test_nested_context_filtering(self):
        """Test filtering with nested context prefixes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            # Only save attention from encoder layer1
            filters = ["encoder.layer1.attention"]

            with ActSave.enabled(save_dir, filters=filters):
                with ActSave.context("encoder"):
                    with ActSave.context("layer1"):
                        ActSave(
                            attention=np.array([1, 2, 3]),  # Should be saved
                            feedforward=np.array([4, 5, 6]),  # Should NOT be saved
                        )

                    with ActSave.context("layer2"):
                        ActSave(
                            attention=np.array([7, 8, 9]),  # Should NOT be saved
                            feedforward=np.array([10, 11, 12]),  # Should NOT be saved
                        )

            activation_dirs = list((save_dir / "0000").iterdir())
            saved_names = {d.name for d in activation_dirs if d.is_dir()}

            expected = {"encoder.layer1.attention"}
            assert saved_names == expected

    def test_wildcard_patterns(self):
        """Test various wildcard patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            # Test different wildcard patterns
            filters = ["*.attention", "layer?_output"]

            with ActSave.enabled(save_dir, filters=filters):
                with ActSave.context("encoder"):
                    ActSave(
                        attention=np.array([1, 2, 3]),  # Matches *.attention
                        feedforward=np.array([4, 5, 6]),  # No match
                    )

                ActSave(
                    layer1_output=np.array([7, 8, 9]),  # Matches layer?_output
                    layer2_output=np.array([10, 11, 12]),  # Matches layer?_output
                    layer10_output=np.array([13, 14, 15]),  # No match (too many chars)
                    other_attention=np.array(
                        [16, 17, 18]
                    ),  # No match for *.attention at root
                )

            activation_dirs = list((save_dir / "0000").iterdir())
            saved_names = {d.name for d in activation_dirs if d.is_dir()}

            expected = {"encoder.attention", "layer1_output", "layer2_output"}
            assert saved_names == expected

    def test_set_filters_method(self):
        """Test the set_filters method for dynamic filter updates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            ActSave.enable(save_dir)

            # Initially no filters
            assert ActSave.filters is None

            # Save some activations (all should be saved)
            ActSave(
                layer1_output=np.array([1, 2, 3]), attention_weights=np.array([4, 5, 6])
            )

            # Set filters and save more activations
            ActSave.set_filters(["layer*"])
            assert ActSave.filters == ["layer*"]

            ActSave(
                layer2_output=np.array([7, 8, 9]),  # Should be saved
                decoder_output=np.array([10, 11, 12]),  # Should NOT be saved
            )

            ActSave.disable()

            activation_dirs = list((save_dir / "0000").iterdir())
            saved_names = {d.name for d in activation_dirs if d.is_dir()}

            expected = {"layer1_output", "attention_weights", "layer2_output"}
            assert saved_names == expected

    def test_filters_property(self):
        """Test the filters property getter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            # Initially no filters
            assert ActSave.filters is None

            # Enable with filters
            filters = ["test*", "layer*"]
            ActSave.enable(save_dir, filters=filters)
            assert ActSave.filters == filters

            # Update filters
            new_filters = ["attention*"]
            ActSave.set_filters(new_filters)
            assert ActSave.filters == new_filters

            # Clear filters
            ActSave.set_filters(None)
            assert ActSave.filters is None

            ActSave.disable()
            assert ActSave.filters is None

    def test_context_manager_with_filters(self):
        """Test the enabled context manager with filters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            filters = ["encoder.*"]

            with ActSave.enabled(save_dir, filters=filters):
                assert ActSave.filters == filters
                assert ActSave.is_enabled

                with ActSave.context("encoder"):
                    ActSave(output=np.array([1, 2, 3]))  # Should be saved

                with ActSave.context("decoder"):
                    ActSave(output=np.array([4, 5, 6]))  # Should NOT be saved

            # After context, filters should be cleared
            assert ActSave.filters is None
            assert not ActSave.is_enabled

            activation_dirs = list((save_dir / "0000").iterdir())
            saved_names = {d.name for d in activation_dirs if d.is_dir()}

            expected = {"encoder.output"}
            assert saved_names == expected

    def test_empty_filters_list(self):
        """Test that empty filters list saves nothing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            # Empty filters list should save nothing
            filters = []

            with ActSave.enabled(save_dir, filters=filters):
                ActSave(
                    layer1_output=np.array([1, 2, 3]),
                    attention_weights=np.array([4, 5, 6]),
                )

            # Should have no saved activations (only the directory structure)
            activation_base_dir = save_dir / "0000"
            if activation_base_dir.exists():
                activation_dirs = list(activation_base_dir.iterdir())
                saved_names = {d.name for d in activation_dirs if d.is_dir()}
                assert saved_names == set()
            else:
                # If no activations were saved, the directory might not exist
                assert True

    def test_filter_with_special_characters(self):
        """Test filters with special characters and edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            # Test character class patterns
            filters = ["layer[12]_*", "*_weights"]

            with ActSave.enabled(save_dir, filters=filters):
                ActSave(
                    layer1_output=np.array([1, 2, 3]),  # Matches layer[12]_*
                    layer2_hidden=np.array([4, 5, 6]),  # Matches layer[12]_*
                    layer3_output=np.array([7, 8, 9]),  # No match
                    attention_weights=np.array([10, 11, 12]),  # Matches *_weights
                    bias_values=np.array([13, 14, 15]),  # No match
                )

            activation_dirs = list((save_dir / "0000").iterdir())
            saved_names = {d.name for d in activation_dirs if d.is_dir()}

            expected = {"layer1_output", "layer2_hidden", "attention_weights"}
            assert saved_names == expected

    def test_filters_persist_across_calls(self):
        """Test that filters persist across multiple ActSave calls."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            filters = ["layer*"]

            with ActSave.enabled(save_dir, filters=filters):
                # First call
                ActSave(layer1_output=np.array([1, 2, 3]))

                # Second call with different activations
                ActSave(
                    layer2_output=np.array([4, 5, 6]),
                    attention=np.array([7, 8, 9]),  # Should NOT be saved
                )

                # Third call
                ActSave(layer3_output=np.array([10, 11, 12]))

            activation_dirs = list((save_dir / "0000").iterdir())
            saved_names = {d.name for d in activation_dirs if d.is_dir()}

            expected = {"layer1_output", "layer2_output", "layer3_output"}
            assert saved_names == expected

    def test_filters_with_disabled_context(self):
        """Test that filters still work when saving is temporarily disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            filters = ["layer*"]

            with ActSave.enabled(save_dir, filters=filters):
                # Normal save
                ActSave(layer1_output=np.array([1, 2, 3]))

                # Disabled context
                with ActSave.disabled():
                    ActSave(
                        layer2_output=np.array(
                            [4, 5, 6]
                        ),  # Should NOT be saved (disabled)
                        attention=np.array([7, 8, 9]),  # Should NOT be saved (disabled)
                    )

                # Re-enabled save
                ActSave(
                    layer3_output=np.array([10, 11, 12]),  # Should be saved
                    decoder_output=np.array(
                        [13, 14, 15]
                    ),  # Should NOT be saved (filtered)
                )

            activation_dirs = list((save_dir / "0000").iterdir())
            saved_names = {d.name for d in activation_dirs if d.is_dir()}

            expected = {"layer1_output", "layer3_output"}
            assert saved_names == expected

    def test_environment_variable_filters(self):
        """Test NSHUTILS_ACTSAVE_FILTERS environment variable parsing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            # Test comma-separated filters
            filters_str = "layer*, attention*"

            with mock.patch.dict(
                os.environ,
                {
                    "NSHUTILS_ACTSAVE": str(save_dir),
                    "NSHUTILS_ACTSAVE_FILTERS": filters_str,
                },
            ):
                # Import a fresh instance to trigger environment variable parsing
                from importlib import reload

                import nshutils.actsave._saver as saver_module

                reload(saver_module)

                # Get the reloaded ActSave instance
                ActSaveEnv = saver_module.ActSave

                try:
                    # Check that filters were parsed correctly
                    assert ActSaveEnv.filters == ["layer*", "attention*"]
                    assert ActSaveEnv.is_enabled

                    # Test that filtering works
                    ActSaveEnv(
                        layer1_output=np.array([1, 2, 3]),  # Should be saved
                        attention_weights=np.array([4, 5, 6]),  # Should be saved
                        decoder_output=np.array([7, 8, 9]),  # Should NOT be saved
                    )

                    # Check what was saved
                    activation_dirs = list((save_dir / "0000").iterdir())
                    saved_names = {d.name for d in activation_dirs if d.is_dir()}

                    expected = {"layer1_output", "attention_weights"}
                    assert saved_names == expected

                finally:
                    ActSaveEnv.disable()

    def test_environment_variable_filters_whitespace_handling(self):
        """Test NSHUTILS_ACTSAVE_FILTERS handles whitespace correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            # Test filters with various whitespace
            filters_str = " layer* , attention* ,  decoder.* "

            with mock.patch.dict(
                os.environ,
                {
                    "NSHUTILS_ACTSAVE": str(save_dir),
                    "NSHUTILS_ACTSAVE_FILTERS": filters_str,
                },
            ):
                from importlib import reload

                import nshutils.actsave._saver as saver_module

                reload(saver_module)

                ActSaveEnv = saver_module.ActSave

                try:
                    # Check that filters were parsed and trimmed correctly
                    assert ActSaveEnv.filters == ["layer*", "attention*", "decoder.*"]

                finally:
                    ActSaveEnv.disable()

    def test_environment_variable_filters_empty_string(self):
        """Test NSHUTILS_ACTSAVE_FILTERS with empty string."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            with mock.patch.dict(
                os.environ,
                {"NSHUTILS_ACTSAVE": str(save_dir), "NSHUTILS_ACTSAVE_FILTERS": ""},
            ):
                from importlib import reload

                import nshutils.actsave._saver as saver_module

                reload(saver_module)

                ActSaveEnv = saver_module.ActSave

                try:
                    # Empty string should result in no filters
                    assert ActSaveEnv.filters is None

                finally:
                    ActSaveEnv.disable()

    def test_environment_variable_filters_only_commas(self):
        """Test NSHUTILS_ACTSAVE_FILTERS with only commas and whitespace."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            with mock.patch.dict(
                os.environ,
                {
                    "NSHUTILS_ACTSAVE": str(save_dir),
                    "NSHUTILS_ACTSAVE_FILTERS": " , , , ",
                },
            ):
                from importlib import reload

                import nshutils.actsave._saver as saver_module

                reload(saver_module)

                ActSaveEnv = saver_module.ActSave

                try:
                    # Should result in no filters
                    assert ActSaveEnv.filters is None

                finally:
                    ActSaveEnv.disable()

    def test_environment_variable_actsave_true_with_filters(self):
        """Test NSHUTILS_ACTSAVE=true with NSHUTILS_ACTSAVE_FILTERS."""
        filters_str = "test*"

        with mock.patch.dict(
            os.environ,
            {"NSHUTILS_ACTSAVE": "true", "NSHUTILS_ACTSAVE_FILTERS": filters_str},
        ):
            from importlib import reload

            import nshutils.actsave._saver as saver_module

            reload(saver_module)

            ActSaveEnv = saver_module.ActSave

            try:
                # Check that it's enabled with filters
                assert ActSaveEnv.is_enabled
                assert ActSaveEnv.filters == ["test*"]

            finally:
                ActSaveEnv.disable()

    def test_enable_when_already_enabled_updates_filters(self):
        """Test that calling enable() when already enabled updates filters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            # First enable with some filters
            ActSave.enable(save_dir, filters=["layer*"])
            assert ActSave.filters == ["layer*"]

            try:
                # Call enable again with different filters
                ActSave.enable(save_dir, filters=["attention*"])

                # Filters should be updated
                assert ActSave.filters == ["attention*"]

                # Test that the new filters work
                ActSave(
                    layer1_output=np.array([1, 2, 3]),  # Should NOT be saved
                    attention_weights=np.array([4, 5, 6]),  # Should be saved
                )

                # Check what was saved
                activation_dirs = list((save_dir / "0000").iterdir())
                saved_names = {d.name for d in activation_dirs if d.is_dir()}

                expected = {"attention_weights"}
                assert saved_names == expected

            finally:
                ActSave.disable()
