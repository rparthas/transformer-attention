"""
Toy datasets for training and validating Transformer implementation.

This module contains:
- Simple vocabulary
- Copy task dataset
- Reverse task dataset
- Collation functions
"""

import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SimpleVocabulary:
    """
    Simple vocabulary for toy tasks.

    Special tokens:
        - <pad>: Padding token (ID = 0)
        - <start>: Start of sequence (ID = 1)
        - <end>: End of sequence (ID = 2)
        - <unk>: Unknown token (ID = 3)

    Regular tokens: Numbers 0 to num_tokens-1
    """

    def __init__(self, num_tokens=100):
        # Special tokens
        self.PAD_TOKEN = '<pad>'
        self.START_TOKEN = '<start>'
        self.END_TOKEN = '<end>'
        self.UNK_TOKEN = '<unk>'

        # Build vocabulary
        special = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]
        numbers = [str(i) for i in range(num_tokens)]

        self.vocab = special + numbers

        # Create mappings
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}

        # Special token IDs
        self.pad_id = self.token2id[self.PAD_TOKEN]
        self.start_id = self.token2id[self.START_TOKEN]
        self.end_id = self.token2id[self.END_TOKEN]
        self.unk_id = self.token2id[self.UNK_TOKEN]

    def encode(self, tokens):
        """Convert tokens to IDs."""
        return [self.token2id.get(token, self.unk_id) for token in tokens]

    def decode(self, ids):
        """Convert IDs to tokens (filter padding)."""
        return [self.id2token[id] for id in ids if id != self.pad_id]

    def __len__(self):
        return len(self.vocab)


class CopyTaskDataset(Dataset):
    """
    Copy task dataset.

    The model learns to copy the input sequence exactly.
    Example: [5, 12, 8] → [5, 12, 8]

    This is the simplest possible seq2seq task and validates basic functionality.

    Args:
        num_samples: Number of samples to generate
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        vocab_size: Vocabulary size
    """

    def __init__(self, num_samples=10000, min_len=3, max_len=10, vocab_size=100):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_size = vocab_size

        # Generate all samples at initialization
        self.samples = self._generate_samples()

    def _generate_samples(self):
        """Generate random sequences to copy."""
        samples = []
        for _ in range(self.num_samples):
            # Random length
            length = random.randint(self.min_len, self.max_len)

            # Random sequence (offset by 4 to skip special tokens)
            sequence = torch.randint(4, self.vocab_size, (length,))

            samples.append(sequence)

        return samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sequence = self.samples[idx]

        # Source: just the sequence
        src = sequence

        # Target: <start> + sequence + <end>
        tgt = torch.cat([
            torch.tensor([1]),  # START_TOKEN id = 1
            sequence,
            torch.tensor([2])   # END_TOKEN id = 2
        ])

        return src, tgt


class ReverseTaskDataset(Dataset):
    """
    Reverse task dataset.

    The model learns to reverse the input sequence.
    Example: [5, 12, 8] → [8, 12, 5]

    Slightly harder than copy task, tests attention mechanism.

    Args:
        num_samples: Number of samples to generate
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        vocab_size: Vocabulary size
    """

    def __init__(self, num_samples=10000, min_len=3, max_len=10, vocab_size=100):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.samples = self._generate_samples()

    def _generate_samples(self):
        samples = []
        for _ in range(self.num_samples):
            length = random.randint(self.min_len, self.max_len)
            sequence = torch.randint(4, self.vocab_size, (length,))
            samples.append(sequence)
        return samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sequence = self.samples[idx]

        # Source: original sequence
        src = sequence

        # Target: <start> + reversed sequence + <end>
        tgt = torch.cat([
            torch.tensor([1]),        # START_TOKEN
            torch.flip(sequence, [0]), # Reverse
            torch.tensor([2])         # END_TOKEN
        ])

        return src, tgt


def collate_fn(batch, pad_id=0):
    """
    Collate function for DataLoader with padding.

    Args:
        batch: List of (src, tgt) tuples
        pad_id: Padding token ID

    Returns:
        srcs_padded: Padded source sequences
        tgts_padded: Padded target sequences
    """
    srcs, tgts = zip(*batch)

    # Pad sources
    src_lens = [len(src) for src in srcs]
    max_src_len = max(src_lens)
    srcs_padded = torch.stack([
        F.pad(src, (0, max_src_len - len(src)), value=pad_id)
        for src in srcs
    ])

    # Pad targets
    tgt_lens = [len(tgt) for tgt in tgts]
    max_tgt_len = max(tgt_lens)
    tgts_padded = torch.stack([
        F.pad(tgt, (0, max_tgt_len - len(tgt)), value=pad_id)
        for tgt in tgts
    ])

    return srcs_padded, tgts_padded
