"""
Training script for toy tasks (copy and reverse).

Usage:
    python train_toy.py --task copy --epochs 10
    python train_toy.py --task reverse --epochs 20
"""

import argparse
import torch
from torch.utils.data import DataLoader

from transformer.model import Transformer
from transformer.datasets import CopyTaskDataset, ReverseTaskDataset, SimpleVocabulary, collate_fn
from transformer.training import create_optimizer, LabelSmoothingLoss, train_epoch
from transformer.inference import evaluate, show_examples


def main():
    parser = argparse.ArgumentParser(description='Train Transformer on toy tasks')
    parser.add_argument('--task', type=str, default='copy', choices=['copy', 'reverse'],
                        help='Task to train on')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--vocab-size', type=int, default=100,
                        help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=128,
                        help='Model dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of encoder/decoder layers')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Warmup steps for learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    print(f"Training Transformer on {args.task} task")
    print(f"Device: {args.device}")
    print(f"Parameters: d_model={args.d_model}, layers={args.num_layers}, heads={args.num_heads}")
    print()

    # Create datasets
    if args.task == 'copy':
        train_dataset = CopyTaskDataset(num_samples=10000, vocab_size=args.vocab_size)
        test_dataset = CopyTaskDataset(num_samples=1000, vocab_size=args.vocab_size)
    else:  # reverse
        train_dataset = ReverseTaskDataset(num_samples=10000, vocab_size=args.vocab_size)
        test_dataset = ReverseTaskDataset(num_samples=1000, vocab_size=args.vocab_size)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Vocabulary
    vocab = SimpleVocabulary(num_tokens=args.vocab_size - 4)  # -4 for special tokens

    # Model
    device = torch.device(args.device)
    model = Transformer(
        src_vocab_size=args.vocab_size,
        tgt_vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_model * 4,
        max_len=50
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")
    print()

    # Optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, args.d_model, args.warmup_steps)

    # Loss function
    criterion = LabelSmoothingLoss(
        vocab_size=args.vocab_size,
        padding_idx=0,
        smoothing=0.1
    )

    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )

        # Evaluate
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}")
        print(f"  Test Acc:   {test_acc:.2%}")

        # Show examples every few epochs
        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            show_examples(model, test_dataset, vocab, device, num_examples=3)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'task': args.task,
                'accuracy': test_acc
            }, f'transformer_{args.task}_toy.pt')

        print()

    print(f"Training complete! Best accuracy: {best_acc:.2%}")
    print(f"Model saved to transformer_{args.task}_toy.pt")


if __name__ == '__main__':
    main()
