"""
Tests for positional encoding.
"""

import pytest
import torch
from transformer.positional import PositionalEncoding, LearnedPositionalEmbedding


class TestPositionalEncoding:
    """Tests for sinusoidal positional encoding."""

    def test_shape(self):
        """Verify encoding has correct shape."""
        pe = PositionalEncoding(d_model=128, max_len=100)
        assert pe.pe.shape == (1, 100, 128)

    def test_deterministic(self):
        """Verify encodings are same across instantiations."""
        pe1 = PositionalEncoding(d_model=128, max_len=100, dropout=0.0)
        pe2 = PositionalEncoding(d_model=128, max_len=100, dropout=0.0)
        assert torch.allclose(pe1.pe, pe2.pe)

    def test_forward_adds_encoding(self):
        """Verify forward pass adds encoding to input."""
        pe = PositionalEncoding(d_model=128, max_len=100, dropout=0.0)
        x = torch.zeros(2, 10, 128)  # (batch, seq, d_model)
        output = pe(x)

        # Output should not be all zeros (encoding was added)
        assert not torch.allclose(output, x)
        assert output.shape == x.shape

    def test_bounded_values(self):
        """Verify encodings are bounded [-1, 1] (sine/cosine)."""
        pe = PositionalEncoding(d_model=128, max_len=100)
        encodings = pe.pe[0]  # (max_len, d_model)

        assert encodings.min() >= -1.0
        assert encodings.max() <= 1.0

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)

        for seq_len in [5, 10, 50]:
            x = torch.randn(2, seq_len, 64)
            output = pe(x)
            assert output.shape == (2, seq_len, 64)

    def test_gradient_flow(self):
        """Test gradients flow through positional encoding."""
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.randn(2, 10, 64, requires_grad=True)

        output = pe(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestLearnedPositionalEmbedding:
    """Tests for learned positional embeddings."""

    def test_shape(self):
        """Verify learned embeddings have correct shape."""
        lpe = LearnedPositionalEmbedding(d_model=128, max_len=100)
        x = torch.randn(2, 10, 128)
        output = lpe(x)

        assert output.shape == (2, 10, 128)

    def test_parameters_trainable(self):
        """Verify embeddings are trainable parameters."""
        lpe = LearnedPositionalEmbedding(d_model=64, max_len=100)

        # Count trainable parameters
        num_params = sum(p.numel() for p in lpe.parameters() if p.requires_grad)
        assert num_params == 100 * 64  # max_len * d_model

    def test_different_from_sinusoidal(self):
        """Verify learned embeddings differ from sinusoidal."""
        sin_pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        learned_pe = LearnedPositionalEmbedding(d_model=64, max_len=100, dropout=0.0)

        x = torch.zeros(1, 10, 64)
        sin_output = sin_pe(x)
        learned_output = learned_pe(x)

        # They should be different (learned starts random)
        assert not torch.allclose(sin_output, learned_output)

    def test_gradient_flow(self):
        """Test gradients flow to embedding parameters."""
        lpe = LearnedPositionalEmbedding(d_model=64, max_len=100, dropout=0.0)
        x = torch.randn(2, 10, 64)

        output = lpe(x)
        loss = output.sum()
        loss.backward()

        # Embedding parameters should have gradients
        assert lpe.embedding.weight.grad is not None
