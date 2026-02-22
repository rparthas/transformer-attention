"""
Training utilities for Transformer model.

This module contains:
- Learning rate scheduler with warmup
- Label smoothing loss
- Training step and epoch functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLRScheduler:
    """
    Learning rate scheduler with warmup and inverse square root decay.

    Formula from paper:
        lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))

    Args:
        optimizer: PyTorch optimizer
        d_model: Model dimension
        warmup_steps: Number of warmup steps (default: 4000)
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self._compute_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _compute_lr(self):
        """Compute learning rate for current step."""
        if self.current_step == 0:
            return 0

        step = self.current_step
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))

        return (self.d_model ** (-0.5)) * min(arg1, arg2)

    def get_lr(self):
        """Get current learning rate."""
        if len(self.optimizer.param_groups) == 0:
            return 0
        return self.optimizer.param_groups[0]['lr']


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross-entropy loss.

    Instead of one-hot targets, distribute some probability mass uniformly.
    This prevents overconfidence and improves generalization.

    Args:
        vocab_size: Size of vocabulary
        padding_idx: Index of padding token (ignored in loss)
        smoothing: Smoothing parameter (default: 0.1)
    """

    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, logits, target):
        """
        Calculate label smoothing loss.

        Args:
            logits: (batch * seq_len, vocab_size) - Model predictions
            target: (batch * seq_len,) - Target labels

        Returns:
            loss: Scalar tensor
        """
        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed target distribution
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # -2 for pad and true label
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0

        # Mask padding
        mask = (target != self.padding_idx)
        true_dist = true_dist * mask.unsqueeze(1)

        # Compute loss
        loss = self.criterion(log_probs, true_dist)

        # Normalize by number of non-padding tokens
        num_tokens = mask.sum()
        if num_tokens > 0:
            return loss / num_tokens
        return loss


def train_step(model, src, tgt, optimizer, scheduler, criterion, pad_idx, device):
    """
    Single training step.

    Args:
        model: Transformer model
        src: (batch, src_len) - Source sequences
        tgt: (batch, tgt_len) - Target sequences
        optimizer: Optimizer
        scheduler: LR scheduler
        criterion: Loss function
        pad_idx: Padding token index
        device: Device (cuda or cpu)

    Returns:
        loss: Scalar loss value
    """
    model.train()
    optimizer.zero_grad()

    # Move to device
    src = src.to(device)
    tgt = tgt.to(device)

    # Forward pass
    # Target input: all except last token
    # Target output: all except first token (<start>)
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    logits = model(src, tgt_input)

    # Reshape for loss calculation
    logits = logits.reshape(-1, logits.size(-1))  # (batch * tgt_len-1, vocab_size)
    tgt_output = tgt_output.reshape(-1)            # (batch * tgt_len-1,)

    # Calculate loss
    loss = criterion(logits, tgt_output)

    # Backward pass
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Update parameters
    optimizer.step()
    scheduler.step()

    return loss.item()


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Train for one epoch.

    Args:
        model: Transformer model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: LR scheduler
        criterion: Loss function
        device: Device

    Returns:
        avg_loss: Average loss for epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (src, tgt) in enumerate(dataloader):
        loss = train_step(model, src, tgt, optimizer, scheduler, criterion, pad_idx=0, device=device)

        total_loss += loss
        num_batches += 1

        # Log progress
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            current_lr = scheduler.get_lr()
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, "
                  f"Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

    return total_loss / num_batches if num_batches > 0 else 0


def create_optimizer(model, d_model, warmup_steps=4000):
    """
    Create Adam optimizer with paper parameters.

    Args:
        model: Model to optimize
        d_model: Model dimension
        warmup_steps: Warmup steps for scheduler

    Returns:
        optimizer: Adam optimizer
        scheduler: LR scheduler
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,  # Will be overridden by scheduler
        betas=(0.9, 0.98),  # β1, β2 from paper
        eps=1e-9            # ε from paper
    )

    scheduler = TransformerLRScheduler(
        optimizer,
        d_model=d_model,
        warmup_steps=warmup_steps
    )

    return optimizer, scheduler
