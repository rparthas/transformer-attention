"""Tests for Transformer model."""

import math
import pytest
import torch
from transformer.model import Encoder, Decoder, Transformer


class TestEncoder:
    """Tests for Encoder."""

    def test_encoder_forward(self):
        """Test complete encoder forward pass."""
        vocab_size = 1000
        encoder = Encoder(
            vocab_size=vocab_size,
            d_model=128,
            num_layers=2,
            num_heads=4,
            d_ff=512
        )

        # Random token sequence
        input_ids = torch.randint(0, vocab_size, (4, 10))  # (batch=4, seq=10)
        output = encoder(input_ids)

        assert output.shape == (4, 10, 128)

    def test_padding_mask_creation(self):
        """Test padding mask is created correctly."""
        encoder = Encoder(vocab_size=1000, d_model=128, num_layers=2)

        # Input with padding (0 = pad token)
        input_ids = torch.tensor([
            [3, 45, 12, 0, 0],
            [67, 89, 0, 0, 0]
        ])

        mask = encoder.create_padding_mask(input_ids)

        # Verify shape
        assert mask.shape == (2, 1, 1, 5)

        # Verify padding positions are masked
        assert mask[0, 0, 0, 0] == 1  # Real token
        assert mask[0, 0, 0, 3] == 0  # Padding
        assert mask[1, 0, 0, 2] == 0  # Padding

    def test_with_explicit_mask(self):
        """Test encoder with provided mask."""
        encoder = Encoder(vocab_size=1000, d_model=128, num_layers=2)
        input_ids = torch.randint(1, 1000, (4, 10))
        mask = torch.ones(4, 1, 1, 10)

        output = encoder(input_ids, mask)
        assert output.shape == (4, 10, 128)

    def test_embedding_scaling(self):
        """Verify embeddings are scaled by sqrt(d_model)."""
        encoder = Encoder(vocab_size=1000, d_model=128, num_layers=2)
        assert encoder.d_model == 128
        assert math.sqrt(encoder.d_model) > 11.3

    def test_different_num_layers(self):
        """Test encoder with different number of layers."""
        for num_layers in [1, 2, 6]:
            encoder = Encoder(
                vocab_size=1000,
                d_model=128,
                num_layers=num_layers,
                num_heads=4,
                d_ff=512
            )

            input_ids = torch.randint(1, 1000, (2, 10))
            output = encoder(input_ids)

            assert output.shape == (2, 10, 128)
            assert len(encoder.layers) == num_layers

    def test_return_attention(self):
        """Test attention weights can be returned."""
        encoder = Encoder(vocab_size=1000, d_model=128, num_layers=2, num_heads=4)
        input_ids = torch.randint(1, 1000, (2, 10))

        output, attn_weights = encoder(input_ids, return_attention=True)

        assert output.shape == (2, 10, 128)
        assert len(attn_weights) == 2  # 2 layers
        assert attn_weights[0].shape == (2, 4, 10, 10)  # (batch, heads, seq, seq)


class TestDecoder:
    """Tests for Decoder."""

    def test_decoder_forward(self):
        """Test complete decoder forward pass."""
        vocab_size = 1000
        decoder = Decoder(
            vocab_size=vocab_size,
            d_model=128,
            num_layers=2,
            num_heads=4,
            d_ff=512
        )

        tgt_ids = torch.randint(1, vocab_size, (4, 10))  # (batch, tgt_seq)
        encoder_output = torch.randn(4, 15, 128)  # (batch, src_seq, d_model)

        logits = decoder(tgt_ids, encoder_output)

        assert logits.shape == (4, 10, vocab_size)

    def test_target_mask_creation(self):
        """Test combined target mask is created correctly."""
        decoder = Decoder(vocab_size=1000, d_model=128, num_layers=2)

        # Target with padding (0 = pad)
        tgt_ids = torch.tensor([
            [2, 45, 12, 0, 0],  # 2=start token, 0=pad
            [2, 67, 89, 23, 0]
        ])

        tgt_mask = decoder.create_target_mask(tgt_ids)

        # Check shape
        assert tgt_mask.shape == (2, 1, 5, 5)

        # Check look-ahead: position 1 can't see position 2
        assert tgt_mask[0, 0, 1, 2] == 0

        # Check padding: position 1 can't see position 3 (padding)
        assert tgt_mask[0, 0, 1, 3] == 0

        # Check allowed: position 1 can see position 0
        assert tgt_mask[0, 0, 1, 0] == 1

    def test_output_shape_varies_with_sequence_length(self):
        """Test decoder handles different sequence lengths."""
        decoder = Decoder(vocab_size=1000, d_model=128, num_layers=2)

        encoder_output = torch.randn(2, 20, 128)

        for tgt_len in [5, 10, 15]:
            tgt_ids = torch.randint(1, 1000, (2, tgt_len))
            logits = decoder(tgt_ids, encoder_output)
            assert logits.shape == (2, tgt_len, 1000)

    def test_with_explicit_masks(self):
        """Test decoder with provided masks."""
        decoder = Decoder(vocab_size=1000, d_model=128, num_layers=2)

        tgt_ids = torch.randint(1, 1000, (2, 10))
        encoder_output = torch.randn(2, 15, 128)

        src_mask = torch.ones(2, 1, 1, 15)
        tgt_mask = torch.tril(torch.ones(10, 10)).unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_mask.expand(2, 1, 10, 10)

        logits = decoder(tgt_ids, encoder_output, src_mask, tgt_mask)
        assert logits.shape == (2, 10, 1000)

    def test_decoder_uses_encoder_output(self):
        """Verify decoder output changes with different encoder outputs."""
        decoder = Decoder(vocab_size=1000, d_model=128, num_layers=2)
        decoder.eval()  # Disable dropout

        tgt_ids = torch.randint(1, 1000, (1, 10))
        encoder_output_1 = torch.randn(1, 15, 128)
        encoder_output_2 = torch.randn(1, 15, 128)

        logits_1 = decoder(tgt_ids, encoder_output_1)
        logits_2 = decoder(tgt_ids, encoder_output_2)

        # Different encoder outputs should produce different logits
        assert not torch.allclose(logits_1, logits_2, atol=1e-4)

    def test_return_attention(self):
        """Test attention weights can be returned."""
        decoder = Decoder(vocab_size=1000, d_model=128, num_layers=2, num_heads=4)
        tgt_ids = torch.randint(1, 1000, (2, 10))
        encoder_output = torch.randn(2, 15, 128)

        logits, (self_attn, cross_attn) = decoder(
            tgt_ids, encoder_output, return_attention=True
        )

        assert logits.shape == (2, 10, 1000)
        assert len(self_attn) == 2  # 2 layers
        assert len(cross_attn) == 2  # 2 layers
        assert self_attn[0].shape == (2, 4, 10, 10)
        assert cross_attn[0].shape == (2, 4, 10, 15)


class TestTransformer:
    """Tests for complete Transformer model."""

    def test_transformer_forward(self):
        """Test end-to-end forward pass."""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            d_ff=512
        )

        src = torch.randint(1, 1000, (4, 15))  # (batch, src_len)
        tgt = torch.randint(1, 1000, (4, 12))  # (batch, tgt_len)

        logits = model(src, tgt)

        assert logits.shape == (4, 12, 1000)

    def test_different_vocab_sizes(self):
        """Test with different source and target vocabularies."""
        model = Transformer(
            src_vocab_size=5000,   # English
            tgt_vocab_size=4000,   # French
            d_model=128,
            num_layers=2
        )

        src = torch.randint(1, 5000, (2, 10))
        tgt = torch.randint(1, 4000, (2, 8))

        logits = model(src, tgt)

        assert logits.shape == (2, 8, 4000)

    def test_encode_decode_separately(self):
        """Test encoding and decoding separately."""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_layers=2
        )

        src = torch.randint(1, 1000, (2, 15))
        tgt = torch.randint(1, 1000, (2, 12))

        # Encode
        encoder_output = model.encode(src)
        assert encoder_output.shape == (2, 15, 128)

        # Decode
        logits = model.decode(tgt, encoder_output)
        assert logits.shape == (2, 12, 1000)

    def test_parameter_count(self):
        """Test parameter counting."""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            d_ff=512
        )

        param_count = model.count_parameters()
        assert param_count > 0

    def test_with_padding(self):
        """Test model handles padding correctly."""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_layers=2,
            src_pad_idx=0,
            tgt_pad_idx=0
        )

        # Source with padding
        src = torch.tensor([
            [5, 10, 15, 20, 0, 0],  # 4 real tokens, 2 padding
            [7, 12, 17, 0, 0, 0]    # 3 real tokens, 3 padding
        ])

        # Target with padding
        tgt = torch.tensor([
            [2, 8, 13, 18, 0],
            [2, 9, 14, 0, 0]
        ])

        logits = model(src, tgt)
        assert logits.shape == (2, 5, 1000)

    def test_gradient_flow(self):
        """Test gradients flow through entire model."""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_layers=2
        )

        src = torch.randint(1, 1000, (2, 10))
        tgt = torch.randint(1, 1000, (2, 8))

        logits = model(src, tgt)
        loss = logits.sum()
        loss.backward()

        # Check some parameters have gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad

    def test_return_attention(self):
        """Test attention weights can be returned."""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4
        )

        src = torch.randint(1, 1000, (2, 10))
        tgt = torch.randint(1, 1000, (2, 8))

        logits, attn = model(src, tgt, return_attention=True)

        assert logits.shape == (2, 8, 1000)
        assert 'encoder_self_attention' in attn
        assert 'decoder_self_attention' in attn
        assert 'decoder_cross_attention' in attn

    def test_xavier_initialization(self):
        """Test that parameters are initialized."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            num_layers=1,
            num_heads=2
        )

        # Check that parameters are not all zeros
        for name, param in model.named_parameters():
            if param.dim() > 1:
                assert not torch.allclose(param, torch.zeros_like(param))
