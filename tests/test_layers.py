"""Tests for Transformer layers."""

import pytest
import torch
import torch.nn as nn
from transformer.layers import (
    PositionwiseFeedForward,
    EncoderLayer,
    DecoderLayer,
    create_look_ahead_mask
)


class TestPositionwiseFeedForward:
    """Tests for Position-wise Feed-Forward Network."""

    def test_output_shape(self):
        """Verify output shape matches input shape."""
        ffn = PositionwiseFeedForward(d_model=128, d_ff=512)
        x = torch.randn(2, 10, 128)  # (batch, seq, d_model)
        output = ffn(x)
        assert output.shape == (2, 10, 128)

    def test_position_independence(self):
        """Verify each position is processed independently."""
        ffn = PositionwiseFeedForward(d_model=128, d_ff=512)
        ffn.eval()  # Set to eval to disable dropout

        x = torch.randn(1, 10, 128)

        # Process full sequence
        full_output = ffn(x)

        # Process each position separately and verify independence
        for i in range(10):
            single_output = ffn(x[:, i:i+1, :])
            assert torch.allclose(
                full_output[:, i:i+1, :],
                single_output,
                atol=1e-6
            )

    def test_dimension_expansion(self):
        """Test internal dimension expansion."""
        ffn = PositionwiseFeedForward(d_model=128, d_ff=512)
        assert ffn.fc1.out_features == 512
        assert ffn.fc2.in_features == 512
        assert ffn.fc2.out_features == 128

    def test_gradient_flow(self):
        """Ensure gradients flow correctly."""
        ffn = PositionwiseFeedForward(d_model=128, d_ff=512)
        x = torch.randn(2, 10, 128, requires_grad=True)
        output = ffn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        ffn = PositionwiseFeedForward(d_model=64, d_ff=256)
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 5, 64)
            output = ffn(x)
            assert output.shape == (batch_size, 5, 64)


class TestEncoderLayer:
    """Tests for Encoder Layer."""

    def test_output_shape(self):
        """Verify output shape matches input."""
        layer = EncoderLayer(d_model=128, num_heads=4, d_ff=512)
        x = torch.randn(2, 10, 128)
        output = layer(x)
        assert output.shape == (2, 10, 128)

    def test_with_mask(self):
        """Test encoder layer with padding mask."""
        layer = EncoderLayer(d_model=128, num_heads=4, d_ff=512)
        x = torch.randn(2, 10, 128)
        # Create mask: first 7 tokens real, last 3 padded
        mask = torch.ones(2, 1, 1, 10)
        mask[:, :, :, 7:] = 0
        output = layer(x, mask)
        assert output.shape == (2, 10, 128)

    def test_gradient_flow(self):
        """Test residual connections enable gradient flow."""
        layer = EncoderLayer(d_model=128, num_heads=4, d_ff=512)
        x = torch.randn(2, 10, 128, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_stacking_layers(self):
        """Test multiple encoder layers can be stacked."""
        layers = nn.ModuleList([
            EncoderLayer(d_model=128, num_heads=4, d_ff=512)
            for _ in range(3)
        ])

        x = torch.randn(2, 10, 128)
        for layer in layers:
            x = layer(x)

        assert x.shape == (2, 10, 128)

    def test_layer_norm_applied(self):
        """Verify layer normalization is working."""
        layer = EncoderLayer(d_model=128, num_heads=4, d_ff=512)
        x = torch.randn(2, 10, 128) * 100  # Large values
        output = layer(x)

        # Output should be normalized (much smaller than input)
        assert output.abs().mean() < x.abs().mean()

    def test_return_attention(self):
        """Test attention weights can be returned."""
        layer = EncoderLayer(d_model=128, num_heads=4, d_ff=512)
        x = torch.randn(2, 10, 128)
        output, attn = layer(x, return_attention=True)
        assert output.shape == (2, 10, 128)
        assert attn.shape == (2, 4, 10, 10)  # (batch, heads, seq, seq)


class TestDecoderLayer:
    """Tests for Decoder Layer."""

    def test_output_shape(self):
        """Verify output shape matches decoder input shape."""
        layer = DecoderLayer(d_model=128, num_heads=4, d_ff=512)
        x = torch.randn(2, 10, 128)  # Decoder input
        enc_output = torch.randn(2, 15, 128)  # Encoder output
        output = layer(x, enc_output)
        assert output.shape == (2, 10, 128)

    def test_with_masks(self):
        """Test decoder layer with both masks."""
        layer = DecoderLayer(d_model=128, num_heads=4, d_ff=512)
        batch, tgt_len, src_len = 2, 10, 15

        x = torch.randn(batch, tgt_len, 128)
        enc_output = torch.randn(batch, src_len, 128)

        # Create masks
        src_mask = torch.ones(batch, 1, 1, src_len)
        tgt_mask = create_look_ahead_mask(tgt_len).unsqueeze(0).unsqueeze(0)

        output = layer(x, enc_output, src_mask, tgt_mask)
        assert output.shape == (batch, tgt_len, 128)

    def test_encoder_decoder_attention_uses_encoder(self):
        """Verify cross-attention uses encoder outputs."""
        layer = DecoderLayer(d_model=128, num_heads=4, d_ff=512)
        layer.eval()  # Disable dropout

        x = torch.randn(1, 5, 128)
        enc_output_1 = torch.randn(1, 5, 128)
        enc_output_2 = torch.randn(1, 5, 128)

        output_1 = layer(x, enc_output_1)
        output_2 = layer(x, enc_output_2)

        # Different encoder outputs should produce different results
        assert not torch.allclose(output_1, output_2, atol=1e-5)

    def test_gradient_flow(self):
        """Test gradients flow to both decoder input and encoder output."""
        layer = DecoderLayer(d_model=128, num_heads=4, d_ff=512)

        x = torch.randn(2, 5, 128, requires_grad=True)
        enc_output = torch.randn(2, 8, 128, requires_grad=True)

        output = layer(x, enc_output)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert enc_output.grad is not None

    def test_return_attention(self):
        """Test attention weights can be returned."""
        layer = DecoderLayer(d_model=128, num_heads=4, d_ff=512)
        x = torch.randn(2, 10, 128)
        enc_output = torch.randn(2, 15, 128)

        output, (self_attn, cross_attn) = layer(
            x, enc_output, return_attention=True
        )

        assert output.shape == (2, 10, 128)
        assert self_attn.shape == (2, 4, 10, 10)  # Self-attention
        assert cross_attn.shape == (2, 4, 10, 15)  # Cross-attention


class TestLookAheadMask:
    """Tests for look-ahead mask creation."""

    def test_look_ahead_mask_shape(self):
        """Test mask has correct shape."""
        mask = create_look_ahead_mask(5)
        assert mask.shape == (5, 5)

    def test_look_ahead_mask_pattern(self):
        """Test look-ahead mask prevents future attention."""
        mask = create_look_ahead_mask(5)

        expected = torch.tensor([
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1]
        ], dtype=torch.float)

        assert torch.equal(mask, expected)

    def test_look_ahead_mask_diagonal(self):
        """Test diagonal is all ones."""
        mask = create_look_ahead_mask(10)
        assert torch.all(torch.diag(mask) == 1)

    def test_look_ahead_mask_upper_triangle(self):
        """Test upper triangle is all zeros."""
        mask = create_look_ahead_mask(10)
        upper = torch.triu(mask, diagonal=1)
        assert torch.all(upper == 0)
