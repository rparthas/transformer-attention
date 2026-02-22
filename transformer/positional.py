"""
Positional encoding for Transformer model.

This module contains:
- Sinusoidal Positional Encoding
- Learned Positional Embeddings (optional)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Why positional encoding?
    Transformers process all positions in parallel (no recurrence), so they need
    explicit position information. Without it, the model would be permutation-invariant.

    Why sine and cosine?
    From paper: "We hypothesized it would allow the model to easily learn to attend
    by relative positions, since for any fixed offset k, PE(pos+k) can be represented
    as a linear function of PE(pos)."

    Args:
        d_model: Model dimension (must match embedding dimension)
        max_len: Maximum sequence length to pre-compute
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create matrix of shape (max_len, d_model) for positional encodings
        pe = torch.zeros(max_len, d_model)

        # Create position indices: [0, 1, 2, ..., max_len-1]
        # Shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Create division term for each dimension
        # div_term = 10000^(2i/d_model) for i=0..d_model/2
        # Using log space for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (saved with model but not trained)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args:
            x: (batch, seq_len, d_model) - Input embeddings

        Returns:
            (batch, seq_len, d_model) - Embeddings + positional encoding
        """
        # Add positional encoding (only up to seq_len)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned Positional Embeddings (alternative to sinusoidal).

    Uses nn.Embedding to learn position representations during training.

    Trade-offs vs Sinusoidal:
    - Pro: May perform better on specific sequence lengths seen in training
    - Con: Cannot extrapolate to longer sequences than seen during training
    - Con: Requires additional parameters to learn

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, x):
        """
        Add learned positional embeddings to input.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # Shape: (1, seq_len)

        # Get position embeddings
        pos_emb = self.embedding(positions)
        # Shape: (1, seq_len, d_model)

        # Add to input
        x = x + pos_emb
        return self.dropout(x)
