"""
Attention mechanisms for Transformer model.

This module contains:
- Scaled Dot-Product Attention
- Multi-Head Attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute Scaled Dot-Product Attention.

    Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Intuition:
    - Q (Query): "What am I looking for?" - current position's information needs
    - K (Key): "What do I contain?" - what each position offers
    - V (Value): "What do I actually provide?" - the information to retrieve
    - QK^T: Compute similarity (dot product) between query and all keys
    - / sqrt(d_k): Scale to prevent extreme values in softmax
    - softmax: Convert scores to probability distribution (sum to 1)
    - multiply V: Weight values by attention probabilities

    Why scaling by sqrt(d_k)?
    From the paper: "We suspect that for large values of d_k, the dot products
    grow large in magnitude, pushing the softmax function into regions where it
    has extremely small gradients."

    Args:
        query: (batch_size, seq_len_q, d_k) - Query tensor
        key: (batch_size, seq_len_k, d_k) - Key tensor
        value: (batch_size, seq_len_v, d_v) - Value tensor (seq_len_v == seq_len_k)
        mask: (batch_size, 1, seq_len_q, seq_len_k) or broadcastable shape
              Positions with mask == 0 will be ignored (set to -inf before softmax)
              Optional, defaults to None (no masking)

    Returns:
        output: (batch_size, seq_len_q, d_v) - Attention output
        attention_weights: (batch_size, seq_len_q, seq_len_k) - Attention probabilities
                          Useful for visualization and interpretability

    Shape examples:
        >>> batch, seq_q, seq_k, d_k = 2, 10, 15, 64
        >>> q = torch.randn(batch, seq_q, d_k)
        >>> k = torch.randn(batch, seq_k, d_k)
        >>> v = torch.randn(batch, seq_k, d_k)
        >>> output, weights = scaled_dot_product_attention(q, k, v)
        >>> output.shape
        torch.Size([2, 10, 64])
        >>> weights.shape
        torch.Size([2, 10, 15])
    """
    # Get d_k from key tensor (last dimension)
    d_k = key.size(-1)

    # Step 1: Compute attention scores (QK^T)
    # query: (batch, seq_q, d_k)
    # key.transpose(-2, -1): (batch, d_k, seq_k)
    # scores: (batch, seq_q, seq_k)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Step 2: Scale by sqrt(d_k) to prevent extreme values
    # This is crucial for stable gradients in softmax
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply mask if provided
    # Masked positions are set to -inf, so after softmax they become 0
    if mask is not None:
        # mask == 0 means "ignore this position"
        # mask == 1 means "attend to this position"
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Apply softmax to get attention probabilities
    # Softmax over last dimension (seq_k): each query position gets a
    # probability distribution over all key positions
    # Shape: (batch, seq_q, seq_k)
    attention_weights = F.softmax(scores, dim=-1)

    # Handle -inf case: if entire row is -inf, softmax returns nan
    # Replace nan with 0 (no attention anywhere)
    attention_weights = torch.nan_to_num(attention_weights)

    # Step 5: Compute weighted sum of values
    # attention_weights: (batch, seq_q, seq_k)
    # value: (batch, seq_k, d_v)
    # output: (batch, seq_q, d_v)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Formula: MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
             where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

    Why multiple heads?
    From the paper: "Multi-head attention allows the model to jointly attend
    to information from different representation subspaces at different positions."

    Intuition:
    - Like having multiple "experts" examine the input simultaneously
    - Each head can specialize in different patterns (syntax, semantics, position, etc.)
    - Heads operate in parallel in lower-dimensional spaces (d_k = d_model / h)
    - Final concatenation combines insights from all heads

    Args:
        d_model: Model dimension (embedding size)
        num_heads: Number of parallel attention heads
        dropout: Dropout probability (default: 0.1)

    Shape:
        Input: (batch, seq_len, d_model)
        Output: (batch, seq_len, d_model)
        Attention weights: (batch, num_heads, seq_len, seq_len)
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        # Validate that d_model is divisible by num_heads
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        # Each projects from d_model to d_model (will be split into heads)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        """
        Split last dimension into (num_heads, d_k).

        Args:
            x: (batch, seq_len, d_model)
            batch_size: batch size

        Returns:
            (batch, num_heads, seq_len, d_k)
        """
        # Reshape: (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        # Transpose: (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x, batch_size):
        """
        Combine heads back into single d_model dimension.

        Args:
            x: (batch, num_heads, seq_len, d_k)
            batch_size: batch size

        Returns:
            (batch, seq_len, d_model)
        """
        # Transpose: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)
        # Reshape: (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        return x.contiguous().view(batch_size, -1, self.d_model)

    def forward(self, query, key, value, mask=None, return_attention=False):
        """
        Forward pass of multi-head attention.

        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            mask: Optional mask (batch, seq_len_q, seq_len_k) or broadcastable
            return_attention: If True, return attention weights

        Returns:
            output: (batch, seq_len_q, d_model)
            attention_weights: (batch, num_heads, seq_len_q, seq_len_k) if return_attention=True
        """
        batch_size = query.size(0)

        # Step 1: Linear projections
        # Q, K, V: (batch, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Step 2: Split into multiple heads
        # Shape: (batch, num_heads, seq_len, d_k)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Step 3: Apply scaled dot-product attention to each head
        # Mask is already shaped (batch, 1, seq_len_q, seq_len_k) or broadcastable
        # No need to unsqueeze again as it will broadcast correctly across heads
        attn_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask
        )
        # attn_output: (batch, num_heads, seq_len_q, d_k)
        # attention_weights: (batch, num_heads, seq_len_q, seq_len_k)

        # Step 4: Concatenate heads
        # (batch, num_heads, seq_len_q, d_k) -> (batch, seq_len_q, d_model)
        concat = self.combine_heads(attn_output, batch_size)

        # Step 5: Apply output projection
        output = self.W_o(concat)

        # Apply dropout
        output = self.dropout(output)

        if return_attention:
            return output, attention_weights
        return output, None
