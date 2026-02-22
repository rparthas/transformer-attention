"""
Encoder and Decoder layers for Transformer model.

This module contains:
- Position-wise Feed-Forward Network
- Encoder Layer
- Decoder Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Formula: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

    What is "position-wise"?
    The same feed-forward network is applied to each position independently
    and identically. This means:
    - No information is shared across positions (unlike attention)
    - Same parameters are used for all positions
    - Each position is processed separately in parallel

    Analogy: Like applying the same function to each element of an array.

    Why two layers?
    1. First layer: Projects to higher dimension (d_ff = 4 * d_model)
       - Creates richer representation space
    2. ReLU: Introduces non-linearity (crucial for learning complex patterns)
    3. Second layer: Projects back to d_model (for residual connection)

    Args:
        d_model: Model dimension (input and output)
        d_ff: Feed-forward inner dimension (typically 4 * d_model)
        dropout: Dropout probability
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        # First linear layer + ReLU: (batch, seq_len, d_model) -> (batch, seq_len, d_ff)
        x = self.fc1(x)
        x = F.relu(x)

        # Dropout for regularization
        x = self.dropout(x)

        # Second linear layer: (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        x = self.fc2(x)

        return x


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer.

    Structure:
        Input
          ↓
        Multi-Head Self-Attention → Dropout → Add & Norm
          ↓
        Position-wise FFN → Dropout → Add & Norm
          ↓
        Output

    Key components:
    1. Multi-head self-attention (tokens attend to all positions)
    2. Position-wise feed-forward network
    3. Residual connections around each sub-layer
    4. Layer normalization after each residual

    Why residual connections?
    - Enable gradient flow in deep networks
    - Model learns residual (what to add) rather than full transformation
    - Easier to learn identity if needed (make sub-layer ≈ 0)

    Why layer normalization?
    - Stabilizes training by normalizing activations
    - Works better than batch norm for NLP (variable sequence lengths)

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward inner dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Sub-layer 1: Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Sub-layer 2: Position-wise feed-forward
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Layer normalization (one for each sub-layer)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attention=False):
        """
        Forward pass through encoder layer.

        Args:
            x: (batch, seq_len, d_model) - Input
            mask: Optional padding mask
            return_attention: If True, return attention weights

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: Optional, if return_attention=True
        """
        # Sub-layer 1: Multi-head self-attention
        attn_output, attn_weights = self.self_attn(
            x, x, x, mask, return_attention=return_attention
        )
        attn_output = self.dropout(attn_output)

        # Residual connection + layer norm
        x = self.norm1(x + attn_output)

        # Sub-layer 2: Position-wise feed-forward
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)

        # Residual connection + layer norm
        x = self.norm2(x + ffn_output)

        if return_attention:
            return x, attn_weights
        return x


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer.

    Structure:
        Decoder Input
          ↓
        Masked Multi-Head Self-Attention → Dropout → Add & Norm
          ↓
        Encoder-Decoder Cross-Attention → Dropout → Add & Norm
          ↓        ↑ (Encoder Output)
        Position-wise FFN → Dropout → Add & Norm
          ↓
        Output

    Three sub-layers:
    1. Masked self-attention: Decoder attends to previous decoder positions
    2. Cross-attention: Decoder attends to encoder output (Q from decoder, K,V from encoder)
    3. Position-wise feed-forward network

    Key difference from encoder:
    - Encoder: 2 sub-layers (self-attention + FFN)
    - Decoder: 3 sub-layers (masked self-attention + cross-attention + FFN)

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward inner dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Sub-layer 1: Masked self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Sub-layer 2: Encoder-decoder cross-attention
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Sub-layer 3: Position-wise feed-forward
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Layer normalization (one for each sub-layer)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None,
                return_attention=False):
        """
        Forward pass through decoder layer.

        Args:
            x: (batch, tgt_seq_len, d_model) - Decoder input
            encoder_output: (batch, src_seq_len, d_model) - Encoder output
            src_mask: Source padding mask
            tgt_mask: Target mask (look-ahead + padding)
            return_attention: If True, return attention weights

        Returns:
            output: (batch, tgt_seq_len, d_model)
            attention_weights: Optional tuple (self_attn, cross_attn) if return_attention=True
        """
        # Sub-layer 1: Masked self-attention
        self_attn_output, self_attn_weights = self.self_attn(
            x, x, x, tgt_mask, return_attention=return_attention
        )
        self_attn_output = self.dropout(self_attn_output)
        x = self.norm1(x + self_attn_output)

        # Sub-layer 2: Encoder-decoder cross-attention
        # Query from decoder, Key and Value from encoder
        cross_attn_output, cross_attn_weights = self.enc_dec_attn(
            x, encoder_output, encoder_output, src_mask,
            return_attention=return_attention
        )
        cross_attn_output = self.dropout(cross_attn_output)
        x = self.norm2(x + cross_attn_output)

        # Sub-layer 3: Position-wise feed-forward
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.norm3(x + ffn_output)

        if return_attention:
            return x, (self_attn_weights, cross_attn_weights)
        return x


def create_look_ahead_mask(size):
    """
    Create look-ahead mask for decoder self-attention.

    Prevents position i from attending to positions j > i.
    This is crucial for auto-regressive generation during training.

    Example for size=5:
        [[1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1]]

    Args:
        size: Sequence length

    Returns:
        mask: (size, size) - Lower triangular matrix
    """
    mask = torch.tril(torch.ones(size, size))
    return mask
