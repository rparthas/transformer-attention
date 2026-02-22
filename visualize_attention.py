"""
Attention Visualization for Transformer Model.

This script demonstrates how to visualize attention weights from a trained model.
Shows encoder self-attention, decoder self-attention, and cross-attention patterns.

Usage:
    python visualize_attention.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from transformer.model import Transformer
from transformer.datasets import SimpleVocabulary


def plot_attention_head(attention, source_tokens, target_tokens,
                        layer_num, head_num, attention_type="encoder"):
    """
    Plot a single attention head as a heatmap.

    Args:
        attention: (seq_len_q, seq_len_k) attention weights
        source_tokens: List of source token strings
        target_tokens: List of target token strings
        layer_num: Layer number
        head_num: Head number
        attention_type: Type of attention (encoder/decoder-self/cross)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(attention, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(range(len(source_tokens)))
    ax.set_yticks(range(len(target_tokens)))
    ax.set_xticklabels(source_tokens, rotation=45, ha='right')
    ax.set_yticklabels(target_tokens)

    # Add colorbar
    plt.colorbar(im, ax=ax)

    ax.set_xlabel('Source Tokens (Key)', fontsize=12)
    ax.set_ylabel('Target Tokens (Query)', fontsize=12)
    ax.set_title(f'{attention_type.title()} - Layer {layer_num}, Head {head_num}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_multihead_attention(attention_weights, source_tokens, target_tokens,
                            layer_num, attention_type="encoder"):
    """
    Plot all attention heads from a single layer in a grid.

    Args:
        attention_weights: (num_heads, seq_len_q, seq_len_k) attention weights
        source_tokens: List of source token strings
        target_tokens: List of target token strings
        layer_num: Layer number
        attention_type: Type of attention
    """
    num_heads = attention_weights.shape[0]
    fig, axes = plt.subplots(2, num_heads // 2, figsize=(20, 10))
    axes = axes.flatten()

    for head in range(num_heads):
        ax = axes[head]
        im = ax.imshow(attention_weights[head], cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        # Only show labels on bottom and left edges
        if head >= num_heads // 2:
            ax.set_xticks(range(len(source_tokens)))
            ax.set_xticklabels(source_tokens, rotation=45, ha='right', fontsize=8)
        else:
            ax.set_xticks([])

        if head % (num_heads // 2) == 0:
            ax.set_yticks(range(len(target_tokens)))
            ax.set_yticklabels(target_tokens, fontsize=8)
        else:
            ax.set_yticks([])

        ax.set_title(f'Head {head + 1}', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax)

    fig.suptitle(f'{attention_type.title()} - Layer {layer_num} - All Heads',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def extract_attention_for_example(model, src_tokens, tgt_tokens, vocab):
    """
    Run model and extract attention weights for a single example.

    Args:
        model: Transformer model
        src_tokens: List of source token IDs
        tgt_tokens: List of target token IDs
        vocab: Vocabulary object

    Returns:
        Dictionary with encoder_attentions, decoder_self_attentions, cross_attentions
    """
    model.eval()

    with torch.no_grad():
        # Add batch dimension
        src = torch.tensor(src_tokens).unsqueeze(0)  # (1, src_len)
        tgt = torch.tensor(tgt_tokens).unsqueeze(0)  # (1, tgt_len)

        # Forward pass with attention extraction
        _, attention_dict = model(src, tgt, return_attention=True)

    # Extract and convert to numpy (remove batch dimension)
    result = {
        'encoder': [attn[0].numpy() for attn in attention_dict['encoder_self_attention']],
        'decoder_self': [attn[0].numpy() for attn in attention_dict['decoder_self_attention']],
        'cross': [attn[0].numpy() for attn in attention_dict['decoder_cross_attention']]
    }

    return result


def demo_attention_patterns():
    """Demonstrate attention visualization with a simple example."""
    print("=" * 70)
    print("ATTENTION VISUALIZATION DEMO")
    print("=" * 70)
    print()

    # Create a simple model
    vocab_size = 50
    d_model = 128
    num_layers = 2
    num_heads = 4

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads
    )
    model.eval()

    # Simple vocabulary
    vocab = SimpleVocabulary(num_tokens=vocab_size - 4)

    # Example sequence (using token IDs)
    # Let's create a simple sequence: "The cat sat on mat"
    src_tokens = [4, 10, 15, 20, 25, 2]  # 2 = EOS
    tgt_tokens = [3, 10, 15, 20, 25, 2]  # 3 = SOS, then copy

    # Token strings for visualization
    src_token_strs = ['The', 'cat', 'sat', 'on', 'mat', '<EOS>']
    tgt_token_strs = ['<SOS>', 'cat', 'sat', 'on', 'mat', '<EOS>']

    print("Example Sequence:")
    print(f"  Source: {' '.join(src_token_strs)}")
    print(f"  Target: {' '.join(tgt_token_strs)}")
    print()

    # Extract attention weights
    print("Extracting attention weights...")
    attentions = extract_attention_for_example(model, src_tokens, tgt_tokens, vocab)

    print(f"  Encoder layers: {len(attentions['encoder'])}")
    print(f"  Decoder layers: {len(attentions['decoder_self'])}")
    print(f"  Cross-attention layers: {len(attentions['cross'])}")
    print()

    # Visualize encoder attention (layer 0, all heads)
    print("Creating visualizations...")

    # 1. Multi-head encoder attention
    enc_attn = attentions['encoder'][0]  # Layer 0: (num_heads, src_len, src_len)
    fig1 = plot_multihead_attention(
        enc_attn, src_token_strs, src_token_strs,
        layer_num=0, attention_type="Encoder Self-Attention"
    )
    fig1.savefig('attention_encoder_multihead.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: attention_encoder_multihead.png")

    # 2. Single head detail (encoder)
    fig2 = plot_attention_head(
        enc_attn[0], src_token_strs, src_token_strs,
        layer_num=0, head_num=0, attention_type="Encoder Self-Attention"
    )
    fig2.savefig('attention_encoder_head0.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: attention_encoder_head0.png")

    # 3. Decoder self-attention
    dec_self_attn = attentions['decoder_self'][0]  # (num_heads, tgt_len, tgt_len)
    fig3 = plot_attention_head(
        dec_self_attn[0], tgt_token_strs, tgt_token_strs,
        layer_num=0, head_num=0, attention_type="Decoder Self-Attention"
    )
    fig3.savefig('attention_decoder_self.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: attention_decoder_self.png")

    # 4. Cross-attention (decoder attending to encoder)
    cross_attn = attentions['cross'][0]  # (num_heads, tgt_len, src_len)
    fig4 = plot_multihead_attention(
        cross_attn, src_token_strs, tgt_token_strs,
        layer_num=0, attention_type="Cross-Attention (Decoder→Encoder)"
    )
    fig4.savefig('attention_cross.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: attention_cross.png")

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print()
    print("🔍 Interpreting the visualizations:")
    print()
    print("1. ENCODER SELF-ATTENTION:")
    print("   - Shows how each source token attends to other source tokens")
    print("   - Diagonal: tokens attending to themselves")
    print("   - Off-diagonal: tokens attending to context")
    print()
    print("2. DECODER SELF-ATTENTION:")
    print("   - Shows causal (left-to-right) attention pattern")
    print("   - Lower triangular: each token only sees previous tokens")
    print("   - Critical for autoregressive generation")
    print()
    print("3. CROSS-ATTENTION:")
    print("   - Shows which source tokens the decoder attends to")
    print("   - Each row: what source tokens influence this target token")
    print("   - Different heads may focus on different relationships")
    print()
    print("4. MULTI-HEAD PATTERNS:")
    print("   - Different heads learn different attention patterns")
    print("   - Some heads: focus on nearby tokens (local context)")
    print("   - Other heads: capture long-range dependencies")
    print()

    # Don't show interactively - images are already saved
    # plt.show()


if __name__ == "__main__":
    demo_attention_patterns()
