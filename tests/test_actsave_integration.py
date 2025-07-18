from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nshutils.actsave import ActLoad, ActSave


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_activations():
    """Sample activation data for testing."""
    return {
        "encoder.layer1.attention": np.random.randn(32, 64),
        "encoder.layer1.feedforward": np.random.randn(32, 128),
        "encoder.layer2.attention": np.random.randn(32, 64),
        "encoder.layer2.feedforward": np.random.randn(32, 128),
        "decoder.layer1.attention": np.random.randn(32, 64),
        "decoder.layer1.feedforward": np.random.randn(32, 128),
        "other.activation": np.random.randn(16, 32),
    }


@pytest.fixture
def populated_activation_dir(temp_dir, sample_activations):
    """Create a directory with saved activations for testing."""
    save_dir = temp_dir / "activations"

    # Enable ActSave and save multiple batches
    ActSave.enable(save_dir=save_dir)

    try:
        # Save 3 batches of activations
        for i in range(3):
            # Add some variation to each batch
            batch_acts = {
                name: arr + np.random.randn(*arr.shape) * 0.1
                for name, arr in sample_activations.items()
            }
            ActSave(batch_acts)

        return save_dir
    finally:
        ActSave.disable()


class TestActLoadBasicFunctionality:
    """Test basic ActLoad functionality."""

    def test_load_from_directory(self, populated_activation_dir):
        """Test loading activations from a directory."""
        loader = ActLoad.from_latest_version(populated_activation_dir)

        # Check that we can access activations
        assert len(loader) > 0
        assert "encoder.layer1.attention" in loader.activations

        # Test accessing specific activation
        attention_acts = loader["encoder.layer1.attention"]
        assert len(attention_acts) == 3  # We saved 3 batches
        assert attention_acts[0].shape == (32, 64)

    def test_load_nonexistent_directory(self, temp_dir):
        """Test that loading from non-existent directory raises appropriate error."""
        nonexistent_dir = temp_dir / "nonexistent"

        with pytest.raises(ValueError, match="not an activation base directory"):
            ActLoad.from_latest_version(nonexistent_dir)

    def test_loaded_activation_indexing(self, populated_activation_dir):
        """Test various indexing methods on LoadedActivation."""
        loader = ActLoad.from_latest_version(populated_activation_dir)
        attention_acts = loader["encoder.layer1.attention"]

        # Test integer indexing
        first_act = attention_acts[0]
        assert isinstance(first_act, np.ndarray)
        assert first_act.shape == (32, 64)

        # Test slice indexing
        slice_acts = attention_acts[0:2]
        assert len(slice_acts) == 2
        assert all(isinstance(act, np.ndarray) for act in slice_acts)

        # Test list indexing
        list_acts = attention_acts[[0, 2]]
        assert len(list_acts) == 2

        # Test iteration
        acts_from_iter = list(attention_acts)
        assert len(acts_from_iter) == 3

    def test_loaded_activation_methods(self, populated_activation_dir):
        """Test LoadedActivation methods."""
        loader = ActLoad.from_latest_version(populated_activation_dir)
        attention_acts = loader["encoder.layer1.attention"]

        # Test len
        assert len(attention_acts) == 3

        # Test all_activations
        all_acts = attention_acts.all_activations()
        assert len(all_acts) == 3
        assert all(isinstance(act, np.ndarray) for act in all_acts)

        # Test repr
        repr_str = repr(attention_acts)
        assert "LoadedActivation" in repr_str
        assert "encoder.layer1.attention" in repr_str
        assert "3 activations" in repr_str


class TestActLoadPrefixFiltering:
    """Test prefix filtering functionality."""

    def test_single_prefix_filter(self, populated_activation_dir):
        """Test filtering by a single prefix."""
        loader = ActLoad.from_latest_version(populated_activation_dir)

        # Filter by encoder prefix
        encoder_loader = loader.filter_by_prefix("encoder.")

        # Check that only encoder activations are present
        encoder_keys = set(encoder_loader.activations.keys())
        expected_keys = {
            "layer1.attention",
            "layer1.feedforward",
            "layer2.attention",
            "layer2.feedforward",
        }
        assert encoder_keys == expected_keys

        # Test that we can access the filtered activations
        attention_acts = encoder_loader["layer1.attention"]
        assert len(attention_acts) == 3
        assert attention_acts[0].shape == (32, 64)

    def test_multiple_prefix_filters_chained(self, populated_activation_dir):
        """Test chaining multiple prefix filters."""
        loader = ActLoad.from_latest_version(populated_activation_dir)

        # Chain filters: encoder -> layer1
        filtered_loader = loader.filter_by_prefix("encoder.").filter_by_prefix(
            "layer1."
        )

        # Should only have attention and feedforward from encoder.layer1
        filtered_keys = set(filtered_loader.activations.keys())
        expected_keys = {"attention", "feedforward"}
        assert filtered_keys == expected_keys

        # Test accessing the deeply filtered activation
        attention_acts = filtered_loader["attention"]
        assert len(attention_acts) == 3
        assert attention_acts[0].shape == (32, 64)

    def test_prefix_filter_no_matches(self, populated_activation_dir):
        """Test prefix filtering with no matches."""
        loader = ActLoad.from_latest_version(populated_activation_dir)

        # Filter by non-existent prefix
        filtered_loader = loader.filter_by_prefix("nonexistent.")

        # Should have no activations
        assert len(filtered_loader) == 0
        assert len(filtered_loader.activations) == 0

    def test_prefix_filter_repr(self, populated_activation_dir):
        """Test that filtered loader has appropriate repr."""
        loader = ActLoad.from_latest_version(populated_activation_dir)

        # Single prefix filter
        encoder_loader = loader.filter_by_prefix("encoder.")
        repr_str = repr(encoder_loader)
        assert "prefix='encoder.'" in repr_str

        # Chained prefix filters
        double_filtered = loader.filter_by_prefix("encoder.").filter_by_prefix(
            "layer1."
        )
        repr_str = repr(double_filtered)
        assert "prefix='encoder.layer1.'" in repr_str

    def test_filtered_loader_get_method(self, populated_activation_dir):
        """Test get method on filtered loader."""
        loader = ActLoad.from_latest_version(populated_activation_dir)
        encoder_loader = loader.filter_by_prefix("encoder.")

        # Test get with existing key
        attention_acts = encoder_loader.get("layer1.attention", None)
        assert attention_acts is not None
        assert len(attention_acts) == 3

        # Test get with non-existing key and default
        nonexistent = encoder_loader.get("nonexistent", "default")
        assert nonexistent == "default"

    def test_filtered_loader_iteration(self, populated_activation_dir):
        """Test iteration over filtered loader."""
        loader = ActLoad.from_latest_version(populated_activation_dir)
        encoder_loader = loader.filter_by_prefix("encoder.")

        # Collect all activations through iteration
        activations_list = list(encoder_loader.values())
        assert len(activations_list) == 4  # 2 layers × 2 types each

        # Verify all are LoadedActivation instances
        from nshutils.actsave._loader import LoadedActivation

        assert all(isinstance(act, LoadedActivation) for act in activations_list)


class TestActLoadContextualActivation:
    """Test ActLoad with context-based activation saving."""

    def test_load_context_saved_activations(self, temp_dir):
        """Test loading activations saved with context."""
        save_dir = temp_dir / "context_activations"

        # Save activations using context
        ActSave.enable(save_dir=save_dir)
        try:
            # Save some activations with context
            with ActSave.context("encoder"):
                ActSave(attention=np.random.randn(32, 64))
                ActSave(feedforward=np.random.randn(32, 128))

            with ActSave.context("decoder"):
                ActSave(attention=np.random.randn(32, 64))
                ActSave(feedforward=np.random.randn(32, 128))
        finally:
            ActSave.disable()

        # Load and test
        loader = ActLoad.from_latest_version(save_dir)

        # Check that context-prefixed names are present
        assert "encoder.attention" in loader.activations
        assert "encoder.feedforward" in loader.activations
        assert "decoder.attention" in loader.activations
        assert "decoder.feedforward" in loader.activations

        # Test filtering by context prefix
        encoder_loader = loader.filter_by_prefix("encoder.")
        assert set(encoder_loader.activations.keys()) == {"attention", "feedforward"}


class TestActLoadVersions:
    """Test ActLoad version handling."""

    def test_multiple_versions(self, temp_dir, sample_activations):
        """Test handling multiple versions of activations."""
        save_dir = temp_dir / "versioned_activations"

        # Create multiple versions
        for version in range(3):
            ActSave.enable(save_dir=save_dir)
            try:
                # Save activations for this version
                batch_acts = {f"version_{version}.data": np.random.randn(10, 10)}
                ActSave(batch_acts)
            finally:
                ActSave.disable()

        # Test all_versions
        all_versions = ActLoad.all_versions(save_dir)
        assert all_versions is not None
        assert len(all_versions) == 3

        # Test from_latest_version loads the most recent
        loader = ActLoad.from_latest_version(save_dir)
        assert "version_2.data" in loader.activations

    def test_is_valid_activation_base(self, temp_dir):
        """Test is_valid_activation_base method."""
        save_dir = temp_dir / "test_base"

        # Initially not valid
        assert not ActLoad.is_valid_activation_base(save_dir)

        # Create activation base
        ActSave.enable(save_dir=save_dir)
        try:
            ActSave(test_data=np.array([1, 2, 3]))
        finally:
            ActSave.disable()

        # Now should be valid
        assert ActLoad.is_valid_activation_base(save_dir)


class TestActLoadErrorHandling:
    """Test error handling in ActLoad."""

    def test_invalid_activation_directory(self, temp_dir):
        """Test handling of invalid activation directories."""
        # Create a fake activation directory without proper structure
        fake_dir = temp_dir / "fake_activation"
        fake_dir.mkdir()

        # Should raise error when trying to create LoadedActivation
        with pytest.raises(ValueError, match="does not exist"):
            from nshutils.actsave._loader import LoadedActivation

            LoadedActivation(fake_dir, "nonexistent_activation")

    def test_filtered_loader_without_base_dir(self, populated_activation_dir):
        """Test that filtered loaders handle activation creation properly."""
        loader = ActLoad.from_latest_version(populated_activation_dir)
        filtered_loader = loader.filter_by_prefix("encoder.")

        # Should be able to create activation using the original base directory
        attention_activation = filtered_loader.activation("layer1.attention")
        assert attention_activation.name == "encoder.layer1.attention"
        assert len(attention_activation) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
