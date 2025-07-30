#!/usr/bin/env python3
"""
Example demonstrating ActSave name filtering functionality.

This example shows how to use the filtering capability to selectively save
only certain activations based on fnmatch patterns. This is different from
the prefix filtering in ActLoad (see actsave_prefix_filtering.py) - this
filters during saving, while prefix filtering filters during loading.

The filtering patterns support fnmatch syntax:
- * matches everything
- ? matches any single character
- [seq] matches any character in seq
- [!seq] matches any character not in seq

Examples:
- "layer*" matches "layer1", "layer2_output", etc.
- "encoder.*" matches "encoder.attention", "encoder.layer1", etc.
- "*.attention" matches "layer1.attention", "decoder.attention", etc.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from nshutils.actsave import ActSave


def basic_filtering_example():
    """Basic example of using filters to save only specific activations."""

    print("=== Basic Filtering Example ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir)

        # Only save activations that match "layer*" or "attention*" patterns
        filters = ["layer*", "attention*"]

        with ActSave.enabled(save_dir, filters=filters):
            print(f"Filters active: {ActSave.filters}")

            # These will be saved (match filters)
            ActSave(
                layer1_output=np.random.randn(32, 64),
                layer2_hidden=np.random.randn(32, 128),
                attention_weights=np.random.randn(8, 32, 32),
            )

            # These will NOT be saved (don't match filters)
            ActSave(
                decoder_output=np.random.randn(32, 256),
                embedding_vector=np.random.randn(32, 512),
                loss_value=np.array(0.42),
            )

        # Only check directory if we created our own save context
        if (
            not ActSave.is_enabled
        ):  # If ActSave was enabled via env vars, skip directory check
            # Check what was actually saved
            activation_dirs = list((save_dir / "0000").iterdir())
            saved_names = sorted([d.name for d in activation_dirs if d.is_dir()])

            print(f"Activations saved: {saved_names}")
            print(f"Expected: ['attention_weights', 'layer1_output', 'layer2_hidden']")
        else:
            print("Activations saved based on filter configuration!")

        print()


def dynamic_filtering_example():
    """Example showing how to update filters dynamically."""

    print("=== Dynamic Filtering Example ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir)

        # Use context manager for clean setup/teardown
        with ActSave.enabled(save_dir):  # Start with no filters
            print(f"Initial filters: {ActSave.filters}")

            # Save some activations (all will be saved)
            ActSave(
                input_embedding=np.random.randn(32, 512),
                layer1_output=np.random.randn(32, 256),
            )

            # Update filters to only save layer outputs
            ActSave.set_filters(["layer*"])
            print(f"Updated filters: {ActSave.filters}")

            # Save more activations (only layer* will be saved)
            ActSave(
                output_embedding=np.random.randn(32, 512),  # Won't be saved
                layer2_output=np.random.randn(32, 128),  # Will be saved
            )

        # Only check directory if we managed our own save context
        if not ActSave.is_enabled:  # If ActSave wasn't enabled via env vars
            # Check what was saved
            activation_dirs = list((save_dir / "0000").iterdir())
            saved_names = sorted([d.name for d in activation_dirs if d.is_dir()])

            print(f"Activations saved: {saved_names}")
            print(f"Expected: ['input_embedding', 'layer1_output', 'layer2_output']")
        else:
            print("Dynamic filtering demonstrated with current ActSave configuration!")

        print()


def contextual_filtering_example():
    """Example showing filtering with context prefixes."""

    print("=== Contextual Filtering Example ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir)

        # Only save activations from encoder layers
        filters = ["encoder.*"]

        with ActSave.enabled(save_dir, filters=filters):
            print(f"Filters active: {ActSave.filters}")

            # Decoder context - these won't be saved
            with ActSave.context("decoder"):
                ActSave(
                    layer1_output=np.random.randn(32, 256),
                    attention=np.random.randn(8, 32, 32),
                )

            # Encoder context - these will be saved
            with ActSave.context("encoder"):
                ActSave(
                    layer1_output=np.random.randn(32, 256),
                    attention=np.random.randn(8, 32, 32),
                )

        # Only check directory if we created our own save context
        original_actsave_enabled = ActSave.is_enabled
        if not original_actsave_enabled:
            # Check what was saved
            activation_dirs = list((save_dir / "0000").iterdir())
            saved_names = sorted([d.name for d in activation_dirs if d.is_dir()])

            print(f"Activations saved: {saved_names}")
            print(f"Expected: ['encoder.attention', 'encoder.layer1_output']")
        else:
            print(
                "Contextual filtering demonstrated with current ActSave configuration!"
            )

        print()


def environment_variable_example():
    """Example showing environment variable configuration."""

    print("=== Environment Variable Example ===")
    print(
        "This example demonstrates how ActSave can be configured via environment variables."
    )
    print()
    print("To test this functionality, run:")
    print(
        '  NSHUTILS_ACTSAVE=1 NSHUTILS_ACTSAVE_FILTERS="layer*,attention*" python examples/actsave_filtering.py'
    )
    print()
    print(f"Current ActSave state:")
    print(f"  - Enabled: {ActSave.is_enabled}")
    print(f"  - Filters: {ActSave.filters}")
    print()

    if ActSave.is_enabled:
        print("ActSave was auto-enabled via environment variables!")

        # Test the current filters
        with tempfile.TemporaryDirectory() as temp_dir:
            # Since ActSave is already enabled, we'll just save some test data
            ActSave(
                layer1_output=np.random.randn(32, 64),
                attention_weights=np.random.randn(8, 32, 32),
                decoder_output=np.random.randn(32, 256),  # This may or may not be saved
            )
            print("Test activations saved based on environment variable configuration!")
    else:
        print("ActSave is not enabled via environment variables.")
        print(
            "Set NSHUTILS_ACTSAVE=1 and optionally NSHUTILS_ACTSAVE_FILTERS=pattern1,pattern2,... to enable it."
        )

    print()


if __name__ == "__main__":
    basic_filtering_example()
    dynamic_filtering_example()
    contextual_filtering_example()
    environment_variable_example()
    print("All filtering examples completed!")
