from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from nshutils.lovely import monkey_patch


@pytest.fixture(autouse=True)
def torch_and_jax_not_installed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda x: None if x in ["torch", "jax"] else True
    )


@pytest.fixture(autouse=True)
def unpatch_numpy_after_test():
    yield
    # Cleanup logic to ensure numpy is in a clean state after each test
    from nshutils.lovely.numpy_ import _np_ge_2

    if _np_ge_2():
        np.set_printoptions(override_repr=None)
    else:
        try:
            # For legacy numpy, reset the string function.
            np.set_string_function(None, True)  # pyright: ignore[reportAttributeAccessIssue]
            np.set_string_function(None, False)  # pyright: ignore[reportAttributeAccessIssue]
        except AttributeError:
            # This might fail if the numpy version is very old and doesn't have it,
            # or if it's numpy 2.0 where it's removed.
            pass


def test_patch_leak():
    # Store the original repr
    original_repr = np.array_repr(np.array([1]))

    # Apply the patch permanently
    patch = monkey_patch(["numpy"])

    # Check that the repr has changed
    assert np.array_repr(np.array([1])) != original_repr

    # Get a weak reference to the patch object to check if it's garbage collected
    patch_ref = weakref.ref(patch)

    # Delete the patch object and trigger garbage collection
    del patch
    gc.collect()

    # The patch object should be garbage collected
    assert patch_ref() is None

    # The patch should still be active
    assert np.array_repr(np.array([1])) != original_repr


def test_patch_context_manager():
    # Store the original repr
    original_repr = np.array_repr(np.array([1]))

    with monkey_patch(["numpy"]):
        # Check that the repr has changed
        assert np.array_repr(np.array([1])) != original_repr

    # Check that the repr has been restored
    assert np.array_repr(np.array([1])) == original_repr


def test_patch_close():
    # Store the original repr
    original_repr = np.array_repr(np.array([1]))

    # Apply the patch
    patch = monkey_patch(["numpy"])
    assert np.array_repr(np.array([1])) != original_repr

    # Close the patch
    patch.close()

    # Check that the repr has been restored
    assert np.array_repr(np.array([1])) == original_repr
