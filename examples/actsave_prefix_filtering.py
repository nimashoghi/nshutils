#!/usr/bin/env python3
"""
Example demonstrating the new prefix filtering functionality in ActLoad.

This example shows how to:
1. Save activations with different prefixes using ActSave
2. Load and filter activations using ActLoad.filter_by_prefix()
3. Chain multiple prefix filters for fine-grained access
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from nshutils.actsave import ActLoad, ActSave


def main():
    # Create a temporary directory for this example
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "example_activations"
        print(f"Saving activations to: {save_dir}")

        # Enable ActSave
        ActSave.enable(save_dir=save_dir)

        try:
            # Simulate saving activations from a transformer model
            # Save 3 batches of data
            for batch_idx in range(3):
                print(f"\nSaving batch {batch_idx + 1}/3...")

                # Save encoder activations
                ActSave(
                    {
                        "encoder.layer1.attention": np.random.randn(32, 64),
                        "encoder.layer1.feedforward": np.random.randn(32, 128),
                        "encoder.layer2.attention": np.random.randn(32, 64),
                        "encoder.layer2.feedforward": np.random.randn(32, 128),
                    }
                )

                # Save decoder activations
                ActSave(
                    {
                        "decoder.layer1.attention": np.random.randn(32, 64),
                        "decoder.layer1.feedforward": np.random.randn(32, 128),
                        "decoder.layer2.attention": np.random.randn(32, 64),
                        "decoder.layer2.feedforward": np.random.randn(32, 128),
                    }
                )

                # Save some other activations
                ActSave(
                    {
                        "output.logits": np.random.randn(32, 1000),
                        "loss.cross_entropy": np.random.randn(32),
                    }
                )

        finally:
            ActSave.disable()

        print(f"\n{'=' * 60}")
        print("LOADING AND FILTERING ACTIVATIONS")
        print(f"{'=' * 60}")

        # Load all activations
        loader = ActLoad.from_latest_version(save_dir)
        print(f"\nAll available activations:")
        for name in sorted(loader.activations.keys()):
            num_acts = len(loader[name])
            print(f"  {name}: {num_acts} activations")

        print(f"\n{'=' * 40}")
        print("BASIC PREFIX FILTERING")
        print(f"{'=' * 40}")

        # Filter by encoder prefix
        encoder_loader = loader.filter_by_prefix("encoder.")
        print(f"\nEncoder activations (after filtering by 'encoder.'):")
        for name in sorted(encoder_loader.activations.keys()):
            num_acts = len(encoder_loader[name])
            print(f"  {name}: {num_acts} activations")

        # Access a specific encoder activation
        attention_acts = encoder_loader["layer1.attention"]
        print(f"\nEncoder layer1 attention activations:")
        print(f"  Shape of first batch: {attention_acts[0].shape}")
        print(f"  Total batches: {len(attention_acts)}")

        print(f"\n{'=' * 40}")
        print("CHAINED PREFIX FILTERING")
        print(f"{'=' * 40}")

        # Chain multiple filters: encoder -> layer1
        layer1_loader = loader.filter_by_prefix("encoder.").filter_by_prefix("layer1.")
        print(
            f"\nEncoder Layer1 activations (chained filtering 'encoder.' -> 'layer1.'):"
        )
        for name in sorted(layer1_loader.activations.keys()):
            num_acts = len(layer1_loader[name])
            print(f"  {name}: {num_acts} activations")

        # Access deeply filtered activation
        feedforward_acts = layer1_loader["feedforward"]
        print(f"\nEncoder layer1 feedforward activations:")
        print(f"  Shape of first batch: {feedforward_acts[0].shape}")
        print(f"  Total batches: {len(feedforward_acts)}")

        print(f"\n{'=' * 40}")
        print("COMPARING DIFFERENT FILTERING APPROACHES")
        print(f"{'=' * 40}")

        # Show different ways to access the same data
        print(f"\nDifferent ways to access encoder.layer2.attention:")

        # Method 1: Direct access
        direct = loader["encoder.layer2.attention"]
        print(
            f"1. Direct: loader['encoder.layer2.attention'] -> {len(direct)} activations"
        )

        # Method 2: Single prefix filter
        single_filter = loader.filter_by_prefix("encoder.")["layer2.attention"]
        print(
            f"2. Single filter: loader.filter_by_prefix('encoder.')['layer2.attention'] -> {len(single_filter)} activations"
        )

        # Method 3: Chained prefix filters
        chained = loader.filter_by_prefix("encoder.").filter_by_prefix("layer2.")[
            "attention"
        ]
        print(
            f"3. Chained filters: ...filter_by_prefix('encoder.').filter_by_prefix('layer2.')['attention'] -> {len(chained)} activations"
        )

        # Verify they all give the same result
        assert np.array_equal(direct[0], single_filter[0])
        assert np.array_equal(direct[0], chained[0])
        print("✓ All methods return identical data!")

        print(f"\n{'=' * 40}")
        print("PRACTICAL USE CASES")
        print(f"{'=' * 40}")

        # Use case 1: Analyze all decoder activations
        print(f"\n1. Analyzing all decoder activations:")
        decoder_loader = loader.filter_by_prefix("decoder.")
        for name, activation in decoder_loader.activations.items():
            mean_val = np.mean([np.mean(act) for act in activation])
            print(f"  decoder.{name}: mean activation = {mean_val:.4f}")

        # Use case 2: Compare attention across layers
        print(f"\n2. Comparing attention patterns across layers:")
        attention_loader = loader.filter_by_prefix("encoder.")
        for name in sorted(attention_loader.activations.keys()):
            if "attention" in name:
                acts = attention_loader[name]
                std_val = np.mean([np.std(act) for act in acts])
                print(f"  encoder.{name}: mean std = {std_val:.4f}")

        print(f"\n{'=' * 60}")
        print("SUCCESS! Prefix filtering example completed.")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
