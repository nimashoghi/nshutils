#!/usr/bin/env python3
"""
Simple test script to verify ActSave filtering functionality.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from nshutils.actsave import ActSave


def test_filtering():
    """Test the filtering functionality of ActSave."""

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir)

        # Test with filters that only save activations matching "layer*"
        filters = ["layer*", "attention*"]

        with ActSave.enabled(save_dir, filters=filters):
            # These should be saved (match filters)
            ActSave(
                layer1_output=np.array([1, 2, 3]),
                layer2_hidden=np.array([4, 5, 6]),
                attention_weights=np.array([0.1, 0.2, 0.3]),
            )

            # These should NOT be saved (don't match filters)
            ActSave(
                decoder_output=np.array([7, 8, 9]),
                embedding_vector=np.array([10, 11, 12]),
            )

        # Check what was actually saved
        activation_dirs = list((save_dir / "0000").iterdir())
        activation_names = [d.name for d in activation_dirs if d.is_dir()]

        print(f"Saved activations: {sorted(activation_names)}")

        # Should only have the filtered activations
        expected = {"layer1_output", "layer2_hidden", "attention_weights"}
        actual = set(activation_names)

        assert expected == actual, f"Expected {expected}, but got {actual}"
        print("✓ Filtering test passed!")


def test_filter_property():
    """Test the filters property and set_filters method."""

    # Test initial state
    assert ActSave.filters is None

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir)

        # Enable with filters
        filters = ["test*"]
        ActSave.enable(save_dir, filters=filters)

        # Check filters property
        assert ActSave.filters == filters

        # Update filters
        new_filters = ["layer*", "attention*"]
        ActSave.set_filters(new_filters)
        assert ActSave.filters == new_filters

        ActSave.disable()
        assert ActSave.filters is None

        print("✓ Filter property test passed!")


if __name__ == "__main__":
    test_filtering()
    test_filter_property()
    print("All tests passed!")
