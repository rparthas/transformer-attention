"""
Test to verify project setup and imports.
"""

import pytest
import torch
import numpy as np
import matplotlib


def test_imports():
    """Test that core libraries are importable."""
    assert torch is not None
    assert np is not None
    assert matplotlib is not None


def test_pytorch_version():
    """Test that PyTorch is properly installed."""
    assert hasattr(torch, '__version__')
    print(f"PyTorch version: {torch.__version__}")


def test_torch_operations():
    """Test basic PyTorch operations work."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    z = x + y

    assert torch.allclose(z, torch.tensor([5.0, 7.0, 9.0]))


def test_numpy_operations():
    """Test basic NumPy operations work."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    z = x + y

    assert np.array_equal(z, np.array([5, 7, 9]))


def test_transformer_package_import():
    """Test that transformer package can be imported."""
    import transformer
    assert transformer is not None
    assert hasattr(transformer, '__version__')
