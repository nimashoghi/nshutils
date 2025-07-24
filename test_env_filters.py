#!/usr/bin/env python3
"""
Demo script to test ACTSAVE_FILTERS environment variable functionality.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from nshutils.actsave import ActSave


def main():
    print("=== Environment Variable Filters Demo ===")
    print(f"ActSave is enabled: {ActSave.is_enabled}")
    print(f"Current filters: {ActSave.filters}")

    if ActSave.is_enabled:
        print("\nTesting filtering with environment variable settings...")

        # Test some activations
        ActSave(
            layer1_output=np.random.randn(32, 64),  # May be saved depending on filters
            layer2_hidden=np.random.randn(32, 128),  # May be saved depending on filters
            attention_weights=np.random.randn(
                8, 32, 32
            ),  # May be saved depending on filters
            decoder_output=np.random.randn(
                32, 256
            ),  # May be saved depending on filters
            embedding_vector=np.random.randn(
                32, 512
            ),  # May be saved depending on filters
        )

        print("Activations saved based on environment variable filters!")
    else:
        print(
            "\nActSave is not enabled. Set ACTSAVE=1 or ACTSAVE=/path/to/save to enable."
        )
        print("Set ACTSAVE_FILTERS=pattern1,pattern2,... to specify filters.")


if __name__ == "__main__":
    main()
