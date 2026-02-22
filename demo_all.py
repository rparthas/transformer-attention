"""
Quick demo showing all implemented components working together.
"""

import torch
from torch.utils.data import DataLoader

from transformer.datasets import CopyTaskDataset, SimpleVocabulary, collate_fn
from transformer.inference import evaluate, greedy_decode
from transformer.model import Transformer
from transformer.training import LabelSmoothingLoss, create_optimizer, train_step

print("=" * 60)
print("Transformer Implementation Demo")
print("=" * 60)

# Configuration
vocab_size = 100
d_model = 64
num_layers = 2
num_heads = 4
batch_size = 8

if torch.backends.mps.is_available():
    device_str = "mps"
    print("mps is used.")
else:
    device_str = "cpu"
    print("cpu is used.")
device = torch.device(device_str)

print(f"\nConfiguration:")
print(f"  vocab_size: {vocab_size}")
print(f"  d_model: {d_model}")
print(f"  num_layers: {num_layers}")
print(f"  num_heads: {num_heads}")
print(f"  device: {device}")

# Create tiny dataset
print(f"\nCreating toy copy dataset...")
train_dataset = CopyTaskDataset(
    num_samples=100, min_len=3, max_len=5, vocab_size=vocab_size
)
test_dataset = CopyTaskDataset(
    num_samples=20, min_len=3, max_len=5, vocab_size=vocab_size
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

# Create model
print(f"\nCreating Transformer model...")
model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_model * 4,
    max_len=20,
).to(device)

print(f"  Model parameters: {model.count_parameters():,}")

# Create optimizer and loss
optimizer, scheduler = create_optimizer(model, d_model, warmup_steps=100)
criterion = LabelSmoothingLoss(vocab_size=vocab_size, padding_idx=0, smoothing=0.1)

# Train for a few steps
print(f"\nTraining for 20 steps...")
for step, (src, tgt) in enumerate(train_loader):
    if step >= 20:
        break
    loss = train_step(
        model, src, tgt, optimizer, scheduler, criterion, pad_idx=0, device=device
    )
    print(f"  Step {step + 1}, Loss: {loss:.4f}, LR: {scheduler.get_lr():.2e}")

# Evaluate
print(f"\nEvaluating...")
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_acc:.2%}")

# Show example
print(f"\nExample prediction:")
src, tgt = test_dataset[0]
vocab = SimpleVocabulary(num_tokens=vocab_size - 4)
src_batch = src.unsqueeze(0)

prediction = greedy_decode(
    model, src_batch, max_len=10, start_id=1, end_id=2, device=device_str
)

src_tokens = vocab.decode(src.tolist())
tgt_tokens = vocab.decode(tgt[1:-1].tolist())
pred_tokens = vocab.decode(prediction[0].tolist()[1:])
if pred_tokens and pred_tokens[-1] == vocab.END_TOKEN:
    pred_tokens = pred_tokens[:-1]

print(f"  Source:    {' '.join(src_tokens)}")
print(f"  Target:    {' '.join(tgt_tokens)}")
print(f"  Predicted: {' '.join(pred_tokens)}")

print(f"\n" + "=" * 60)
print("Demo complete! All components working.")
print("=" * 60)
