"""
Inference utilities for Transformer model.

This module contains:
- Greedy decoding
- Evaluation functions
"""

import torch
import torch.nn.functional as F


def greedy_decode(model, src, max_len, start_id, end_id, device='cpu'):
    """
    Generate sequence using greedy decoding.

    Args:
        model: Transformer model
        src: (batch, src_len) - Source sequence
        max_len: Maximum generation length
        start_id: Start token ID
        end_id: End token ID
        device: Device

    Returns:
        tgt: (batch, generated_len) - Generated sequence
    """
    model.eval()
    src = src.to(device)

    # Encode source
    encoder_output = model.encode(src)

    # Start with start token
    batch_size = src.size(0)
    tgt = torch.full((batch_size, 1), start_id, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len):
            # Decode
            logits = model.decode(tgt, encoder_output)

            # Get next token (greedy)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)

            # Stop if all sequences have generated end token
            if torch.all(next_token == end_id):
                break

    return tgt


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on test set.

    Args:
        model: Transformer model
        dataloader: Test data loader
        criterion: Loss function
        device: Device

    Returns:
        avg_loss: Average loss
        accuracy: Token-level accuracy
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            # Forward pass
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)

            # Calculate loss
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt_output.reshape(-1)
            loss = criterion(logits_flat, tgt_flat)
            total_loss += loss.item()

            # Calculate accuracy (ignoring padding)
            predictions = logits.argmax(dim=-1)
            mask = (tgt_output != 0)  # Non-padding
            correct = (predictions == tgt_output) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    return avg_loss, accuracy


def show_examples(model, dataset, vocab, device, num_examples=5):
    """
    Show example predictions.

    Args:
        model: Transformer model
        dataset: Dataset to sample from
        vocab: Vocabulary for decoding
        device: Device
        num_examples: Number of examples to show
    """
    model.eval()

    print("\nExample Predictions:")
    print("-" * 60)

    for i in range(min(num_examples, len(dataset))):
        src, tgt = dataset[i]
        src_batch = src.unsqueeze(0).to(device)

        # Greedy decoding
        with torch.no_grad():
            prediction = greedy_decode(
                model, src_batch, max_len=20,
                start_id=vocab.start_id,
                end_id=vocab.end_id,
                device=device
            )

        # Decode
        src_tokens = vocab.decode(src.tolist())
        tgt_tokens = vocab.decode(tgt[1:-1].tolist())  # Remove start/end
        pred_tokens = vocab.decode(prediction[0].cpu().tolist()[1:])  # Remove start

        # Remove end token if present
        if pred_tokens and pred_tokens[-1] == vocab.END_TOKEN:
            pred_tokens = pred_tokens[:-1]

        print(f"Source:    {' '.join(src_tokens)}")
        print(f"Target:    {' '.join(tgt_tokens)}")
        print(f"Predicted: {' '.join(pred_tokens)}")
        match = '\u2713' if pred_tokens == tgt_tokens else '\u2717'
        print(f"Match: {match}")
        print()
