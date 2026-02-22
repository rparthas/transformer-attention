"""
Tests for attention mechanisms.
"""

import pytest
import torch
from transformer.attention import scaled_dot_product_attention, MultiHeadAttention


class TestScaledDotProductAttention:
    """Tests for scaled dot-product attention function."""

    def test_output_shape(self):
        """Verify output has correct shape."""
        batch_size, seq_len_q, seq_len_k, d_k = 2, 10, 15, 64

        query = torch.randn(batch_size, seq_len_q, d_k)
        key = torch.randn(batch_size, seq_len_k, d_k)
        value = torch.randn(batch_size, seq_len_k, d_k)

        output, attention_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch_size, seq_len_q, d_k)
        assert attention_weights.shape == (batch_size, seq_len_q, seq_len_k)

    def test_attention_weights_normalized(self):
        """Verify attention weights sum to 1 along last dimension."""
        batch_size, seq_len, d_k = 2, 5, 32

        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_k)

        _, attention_weights = scaled_dot_product_attention(query, key, value)

        # Sum over seq_k dimension (last dim) should be 1.0
        sums = attention_weights.sum(dim=-1)
        expected = torch.ones(batch_size, seq_len)

        assert torch.allclose(sums, expected, atol=1e-6)

    def test_self_attention_simple_case(self):
        """Test self-attention with identical Q, K, V."""
        batch_size, seq_len, d_k = 1, 3, 4

        # Use same tensor for Q, K, V (self-attention)
        x = torch.randn(batch_size, seq_len, d_k)

        output, weights = scaled_dot_product_attention(x, x, x)

        # Check shapes
        assert output.shape == (batch_size, seq_len, d_k)
        assert weights.shape == (batch_size, seq_len, seq_len)

        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, seq_len))

    def test_masking_prevents_attention(self):
        """Verify masking correctly prevents attention to masked positions."""
        batch_size, seq_len, d_k = 1, 4, 8

        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_k)

        # Create mask: allow attention to first 2 positions, mask last 2
        # Mask shape should match scores shape for no broadcasting issues
        mask = torch.ones(batch_size, seq_len, seq_len)
        mask[:, :, 2:] = 0  # Mask key positions 2 and 3

        output, attention_weights = scaled_dot_product_attention(
            query, key, value, mask
        )

        # Attention weights for key positions 2 and 3 should be 0
        # Shape: attention_weights is (batch, seq_q, seq_k)
        assert torch.allclose(attention_weights[:, :, 2:], torch.zeros(batch_size, seq_len, 2))

        # Attention weights for key positions 0 and 1 should sum to 1
        active_weights = attention_weights[:, :, :2]
        assert torch.allclose(active_weights.sum(dim=-1), torch.ones(batch_size, seq_len))

    def test_look_ahead_mask(self):
        """Test causal (look-ahead) masking for decoder."""
        batch_size, seq_len, d_k = 1, 4, 8

        query = torch.randn(batch_size, seq_len, d_k)
        key = query.clone()
        value = query.clone()

        # Create lower triangular mask (causal mask)
        # Position i can only attend to positions <= i
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        # Shape: (1, seq_len, seq_len)

        _, attention_weights = scaled_dot_product_attention(
            query, key, value, mask
        )

        # Check that attention to future positions is 0
        # The attention_weights shape is (batch, seq_q, seq_k)
        # Position 0 should only attend to position 0
        assert attention_weights[0, 0, 1:].sum().item() < 1e-6

        # Position 1 should only attend to positions 0 and 1
        assert attention_weights[0, 1, 2:].sum().item() < 1e-6

        # Position 2 should only attend to positions 0, 1, 2
        assert attention_weights[0, 2, 3:].sum().item() < 1e-6

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths for Q and K."""
        batch_size, seq_q, seq_k, d_k = 2, 10, 20, 32

        query = torch.randn(batch_size, seq_q, d_k)
        key = torch.randn(batch_size, seq_k, d_k)
        value = torch.randn(batch_size, seq_k, d_k)

        output, attention_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch_size, seq_q, d_k)
        assert attention_weights.shape == (batch_size, seq_q, seq_k)

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        batch_size, seq_len, d_k = 1, 5, 16

        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_k)

        output, attention_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (1, 5, 16)
        assert attention_weights.shape == (1, 5, 5)

    def test_gradient_flow(self):
        """Verify gradients flow correctly through attention."""
        batch_size, seq_len, d_k = 2, 5, 16

        query = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        key = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
        value = torch.randn(batch_size, seq_len, d_k, requires_grad=True)

        output, _ = scaled_dot_product_attention(query, key, value)

        # Compute dummy loss and backpropagate
        loss = output.sum()
        loss.backward()

        # All inputs should have gradients
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None

        # Gradients should not be all zeros
        assert not torch.allclose(query.grad, torch.zeros_like(query.grad))
        assert not torch.allclose(key.grad, torch.zeros_like(key.grad))
        assert not torch.allclose(value.grad, torch.zeros_like(value.grad))

    def test_small_d_k(self):
        """Test with small d_k value."""
        batch_size, seq_len, d_k = 2, 3, 4

        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_k)

        output, attention_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch_size, seq_len, d_k)
        assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(batch_size, seq_len))

    def test_large_d_k(self):
        """Test with large d_k value (benefits most from scaling)."""
        batch_size, seq_len, d_k = 2, 5, 512

        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_k)

        output, attention_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch_size, seq_len, d_k)
        assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(batch_size, seq_len))

    def test_sequence_length_one(self):
        """Test edge case with sequence length of 1."""
        batch_size, seq_len, d_k = 2, 1, 16

        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_k)

        output, attention_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch_size, 1, d_k)
        assert attention_weights.shape == (batch_size, 1, 1)
        # With seq_len=1, attention weight should be 1.0
        assert torch.allclose(attention_weights, torch.ones(batch_size, 1, 1))

    def test_all_masked(self):
        """Test case where all positions are masked."""
        batch_size, seq_len, d_k = 1, 4, 8

        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_k)

        # Mask everything
        mask = torch.zeros(batch_size, 1, seq_len, seq_len)

        output, attention_weights = scaled_dot_product_attention(
            query, key, value, mask
        )

        # All attention weights should be 0 (handled by nan_to_num)
        assert torch.allclose(attention_weights, torch.zeros(batch_size, seq_len, seq_len))

        # Output should be zeros (no attention anywhere)
        assert torch.allclose(output, torch.zeros(batch_size, seq_len, d_k))


class TestMultiHeadAttention:
    """Tests for Multi-Head Attention class."""

    def test_initialization_valid(self):
        """Test valid initialization (d_model divisible by num_heads)."""
        mha = MultiHeadAttention(d_model=128, num_heads=4)
        assert mha.d_k == 32
        assert mha.num_heads == 4
        assert mha.d_model == 128

    def test_initialization_invalid(self):
        """Test that invalid num_heads raises error."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=128, num_heads=5)  # Not divisible

    def test_forward_shape_preserved(self):
        """Test output shape matches input shape."""
        mha = MultiHeadAttention(d_model=128, num_heads=4)
        x = torch.randn(2, 10, 128)  # (batch, seq, d_model)

        output, _ = mha(x, x, x)

        assert output.shape == (2, 10, 128)

    def test_attention_weights_shape(self):
        """Test attention weights have correct shape for all heads."""
        mha = MultiHeadAttention(d_model=128, num_heads=4)
        x = torch.randn(2, 10, 128)

        _, attn_weights = mha(x, x, x, return_attention=True)

        # Shape: (batch, num_heads, seq_len, seq_len)
        assert attn_weights.shape == (2, 4, 10, 10)

    def test_different_qkv_lengths(self):
        """Test with different sequence lengths for Q, K, V."""
        mha = MultiHeadAttention(d_model=64, num_heads=4)

        query = torch.randn(2, 10, 64)  # seq_len = 10
        key = torch.randn(2, 15, 64)     # seq_len = 15
        value = torch.randn(2, 15, 64)   # seq_len = 15

        output, attn_weights = mha(query, key, value, return_attention=True)

        assert output.shape == (2, 10, 64)  # Matches query seq_len
        assert attn_weights.shape == (2, 4, 10, 15)  # (batch, heads, seq_q, seq_k)

    def test_with_mask(self):
        """Test multi-head attention with mask."""
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        x = torch.randn(2, 10, 64)

        # Create causal mask
        mask = torch.tril(torch.ones(10, 10)).unsqueeze(0)  # (1, 10, 10)

        output, _ = mha(x, x, x, mask=mask)

        assert output.shape == (2, 10, 64)

    def test_gradient_flow(self):
        """Test gradients flow through all heads."""
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        x = torch.randn(2, 10, 64, requires_grad=True)

        output, _ = mha(x, x, x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_batch_size_variations(self):
        """Test with different batch sizes."""
        mha = MultiHeadAttention(d_model=128, num_heads=4)

        for batch_size in [1, 2, 8]:
            x = torch.randn(batch_size, 10, 128)
            output, _ = mha(x, x, x)
            assert output.shape == (batch_size, 10, 128)

    def test_return_attention_flag(self):
        """Test return_attention flag works correctly."""
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        x = torch.randn(2, 10, 64)

        # Without return_attention
        output, attn = mha(x, x, x, return_attention=False)
        assert output.shape == (2, 10, 64)
        assert attn is None

        # With return_attention
        output, attn = mha(x, x, x, return_attention=True)
        assert output.shape == (2, 10, 64)
        assert attn is not None
        assert attn.shape == (2, 4, 10, 10)
