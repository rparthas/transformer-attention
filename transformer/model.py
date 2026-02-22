"""
Complete Transformer model.

This module contains:
- Encoder (stack of encoder layers)
- Decoder (stack of decoder layers)
- Full Transformer (encoder + decoder)
"""

import math

import torch
import torch.nn as nn

from .layers import DecoderLayer, EncoderLayer
from .positional import PositionalEncoding


class Encoder(nn.Module):
    """
    Transformer Encoder.

    Stack of N encoder layers with:
    - Token embeddings
    - Positional encoding
    - Multiple encoder layers (self-attention + FFN)

    Args:
        vocab_size: Size of source vocabulary
        d_model: Model dimension
        num_layers: Number of encoder layers
        num_heads: Number of attention heads
        d_ff: Feed-forward inner dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        pad_token_id: Padding token ID
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
        max_len=1000,
        dropout=0.1,
        pad_token_id=0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Stack of encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, input_ids, mask=None, return_attention=False):
        """
        Forward pass through encoder.

        Args:
            input_ids: (batch, seq_len) - Token indices
            mask: Optional padding mask
            return_attention: If True, return attention weights from all layers

        Returns:
            output: (batch, seq_len, d_model) - Encoded representations
            attention_weights: Optional list of attention weights if return_attention=True
        """
        # Create mask if not provided
        if mask is None:
            mask = self.create_padding_mask(input_ids)

        # Token embeddings with scaling (paper implementation)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through encoder layers
        attention_weights = []
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x, mask, return_attention=True)
                attention_weights.append(attn)
            else:
                x = layer(x, mask)

        if return_attention:
            return x, attention_weights
        return x

    def create_padding_mask(self, input_ids):
        """
        Create padding mask from input token IDs.

        Args:
            input_ids: (batch, seq_len)

        Returns:
            mask: (batch, 1, 1, seq_len)
        """
        mask = (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask.float()


class Decoder(nn.Module):
    """
    Transformer Decoder.

    Stack of N decoder layers with:
    - Target embeddings
    - Positional encoding
    - Multiple decoder layers (masked self-attention + cross-attention + FFN)
    - Output projection to vocabulary

    Args:
        vocab_size: Size of target vocabulary
        d_model: Model dimension
        num_layers: Number of decoder layers
        num_heads: Number of attention heads
        d_ff: Feed-forward inner dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        pad_token_id: Padding token ID
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
        max_len=1000,
        dropout=0.1,
        pad_token_id=0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # Target embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt_ids,
        encoder_output,
        src_mask=None,
        tgt_mask=None,
        return_attention=False,
    ):
        """
        Forward pass through decoder.

        Args:
            tgt_ids: (batch, tgt_seq_len) - Target token indices
            encoder_output: (batch, src_seq_len, d_model) - Encoder output
            src_mask: Source padding mask
            tgt_mask: Target mask (look-ahead + padding)
            return_attention: If True, return attention weights

        Returns:
            logits: (batch, tgt_seq_len, vocab_size) - Output logits
            attention_weights: Optional tuple (self_attn_list, cross_attn_list)
        """
        # Create target mask if not provided
        if tgt_mask is None:
            tgt_mask = self.create_target_mask(tgt_ids)

        # Target embeddings with scaling
        x = self.embedding(tgt_ids) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through decoder layers
        self_attention_weights = []
        cross_attention_weights = []

        for layer in self.layers:
            if return_attention:
                x, (self_attn, cross_attn) = layer(
                    x, encoder_output, src_mask, tgt_mask, return_attention=True
                )
                self_attention_weights.append(self_attn)
                cross_attention_weights.append(cross_attn)
            else:
                x = layer(x, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary
        logits = self.fc_out(x)

        if return_attention:
            return logits, (self_attention_weights, cross_attention_weights)
        return logits

    def create_target_mask(self, tgt_ids):
        """
        Create combined look-ahead and padding mask for target.

        Args:
            tgt_ids: (batch, tgt_len)

        Returns:
            mask: (batch, 1, tgt_len, tgt_len)
        """
        batch_size, tgt_len = tgt_ids.size()

        # Padding mask: (batch, 1, 1, tgt_len)
        tgt_padding_mask = (tgt_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)

        # Look-ahead mask: (1, 1, tgt_len, tgt_len)
        look_ahead_mask = (
            torch.tril(torch.ones(tgt_len, tgt_len, device=tgt_ids.device))
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Combine (both must be True for attention to be allowed)
        tgt_mask = tgt_padding_mask & look_ahead_mask.bool()

        return tgt_mask.float()


class Transformer(nn.Module):
    """
    Complete Transformer model.

    Combines encoder and decoder for sequence-to-sequence tasks.

    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension
        num_layers: Number of encoder/decoder layers
        num_heads: Number of attention heads
        d_ff: Feed-forward inner dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        src_pad_idx: Source padding token index
        tgt_pad_idx: Target padding token index
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
        max_len=1000,
        dropout=0.1,
        src_pad_idx=0,
        tgt_pad_idx=0,
    ):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.d_model = d_model

        # Encoder
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pad_token_id=src_pad_idx,
        )

        # Decoder
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pad_token_id=tgt_pad_idx,
        )

        # Initialize weights
        self._init_weights()

    def forward(self, src, tgt, return_attention=False):
        """
        Forward pass through transformer.

        Args:
            src: (batch, src_seq_len) - Source token IDs
            tgt: (batch, tgt_seq_len) - Target token IDs
            return_attention: If True, return attention weights

        Returns:
            logits: (batch, tgt_seq_len, tgt_vocab_size)
            attention_weights: Optional dict with encoder/decoder attention
        """
        # Create masks
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)

        # Encode source
        if return_attention:
            encoder_output, enc_attn = self.encoder(
                src, src_mask, return_attention=True
            )
        else:
            encoder_output = self.encoder(src, src_mask)

        # Decode target
        if return_attention:
            logits, (dec_self_attn, dec_cross_attn) = self.decoder(
                tgt, encoder_output, src_mask, tgt_mask, return_attention=True
            )
            attention_weights = {
                "encoder_self_attention": enc_attn,
                "decoder_self_attention": dec_self_attn,
                "decoder_cross_attention": dec_cross_attn,
            }
            return logits, attention_weights
        else:
            logits = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        return logits

    def create_src_mask(self, src):
        """Create padding mask for source sequence."""
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def create_tgt_mask(self, tgt):
        """Create combined padding and look-ahead mask for target."""
        batch_size, tgt_len = tgt.size()

        # Padding mask
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)

        # Look-ahead mask
        tgt_sub_mask = (
            torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device))
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Combine
        tgt_mask = tgt_pad_mask & tgt_sub_mask.bool()
        return tgt_mask.float()

    def encode(self, src, src_mask=None, return_attention=False):
        """Encode source sequence (useful for inference)."""
        if src_mask is None:
            src_mask = self.create_src_mask(src)
        return self.encoder(src, src_mask, return_attention=return_attention)

    def decode(
        self, tgt, encoder_output, src_mask=None, tgt_mask=None, return_attention=False
    ):
        """Decode target sequence given encoder output."""
        if tgt_mask is None:
            tgt_mask = self.create_tgt_mask(tgt)
        return self.decoder(
            tgt, encoder_output, src_mask, tgt_mask, return_attention=return_attention
        )

    def _init_weights(self):
        """Initialize parameters using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
