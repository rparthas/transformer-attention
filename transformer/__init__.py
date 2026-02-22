"""
Transformer: Educational PyTorch Implementation

A toy implementation of "Attention Is All You Need" for learning purposes.
Focus on clarity and understanding over performance.
"""

__version__ = "0.1.0"

from .attention import scaled_dot_product_attention, MultiHeadAttention
from .positional import PositionalEncoding, LearnedPositionalEmbedding
from .layers import (
    PositionwiseFeedForward,
    EncoderLayer,
    DecoderLayer,
    create_look_ahead_mask
)
from .model import Encoder, Decoder, Transformer
from .training import (
    TransformerLRScheduler,
    LabelSmoothingLoss,
    train_step,
    train_epoch,
    create_optimizer
)
from .datasets import (
    SimpleVocabulary,
    CopyTaskDataset,
    ReverseTaskDataset,
    collate_fn
)
from .inference import (
    greedy_decode,
    evaluate,
    show_examples
)

__all__ = [
    # Attention mechanisms
    'scaled_dot_product_attention',
    'MultiHeadAttention',
    # Positional encoding
    'PositionalEncoding',
    'LearnedPositionalEmbedding',
    # Layers
    'PositionwiseFeedForward',
    'EncoderLayer',
    'DecoderLayer',
    'create_look_ahead_mask',
    # Model
    'Encoder',
    'Decoder',
    'Transformer',
    # Training
    'TransformerLRScheduler',
    'LabelSmoothingLoss',
    'train_step',
    'train_epoch',
    'create_optimizer',
    # Datasets
    'SimpleVocabulary',
    'CopyTaskDataset',
    'ReverseTaskDataset',
    'collate_fn',
    # Inference
    'greedy_decode',
    'evaluate',
    'show_examples',
]
