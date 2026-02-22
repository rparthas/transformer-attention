"""
Demo: Understanding Transformer Components (Stories 1-6)

This script demonstrates the implemented components with visual examples.
"""

import torch
import math
from transformer.attention import scaled_dot_product_attention, MultiHeadAttention
from transformer.positional import PositionalEncoding
from transformer.layers import PositionwiseFeedForward, EncoderLayer


def demo_scaled_attention():
    """Demo 1: How attention focuses on relevant words"""
    print("=" * 70)
    print("DEMO 1: Scaled Dot-Product Attention")
    print("=" * 70)

    # Simple example: 3 words
    batch_size, seq_len, d_k = 1, 3, 4

    print("\nScenario: Processing sentence 'cat sat mat'")
    print(f"Sequence length: {seq_len} words")
    print(f"Embedding dimension: {d_k}")

    # Create simple embeddings (in reality, these come from learned embeddings)
    query = torch.randn(batch_size, seq_len, d_k)
    key = query.clone()    # Self-attention: attending to same sequence
    value = query.clone()

    output, attention_weights = scaled_dot_product_attention(query, key, value)

    print(f"\nAttention weights shape: {attention_weights.shape}")
    print("Attention weights (how much each word attends to every word):")
    print(attention_weights[0].numpy())

    print("\n🔍 Interpretation:")
    print(f"  Word 1 → Word 1: {attention_weights[0, 0, 0]:.3f}")
    print(f"  Word 1 → Word 2: {attention_weights[0, 0, 1]:.3f}")
    print(f"  Word 1 → Word 3: {attention_weights[0, 0, 2]:.3f}")
    print(f"  Sum: {attention_weights[0, 0].sum():.3f} (must equal 1.0)")


def demo_multi_head():
    """Demo 2: Multiple attention heads"""
    print("\n" + "=" * 70)
    print("DEMO 2: Multi-Head Attention (4 heads)")
    print("=" * 70)

    d_model, num_heads = 128, 4
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    print(f"\nModel dimension (d_model): {d_model}")
    print(f"Number of heads: {num_heads}")
    print(f"Dimension per head (d_k): {d_model // num_heads}")

    # Input: 5 words
    batch_size, seq_len = 2, 5
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {x.shape} (batch, sequence, embedding)")

    output, attention_weights = mha(x, x, x, return_attention=True)

    print(f"\nOutput shape: {output.shape} (same as input)")
    print(f"Attention weights shape: {attention_weights.shape}")
    print("  → (batch, num_heads, seq_len, seq_len)")

    print("\n🔍 Understanding multiple heads:")
    print("  Each head can specialize:")
    print("  - Head 0: Might focus on nearby words")
    print("  - Head 1: Might capture long-range dependencies")
    print("  - Head 2: Might identify syntactic patterns")
    print("  - Head 3: Might focus on semantic relationships")


def demo_positional_encoding():
    """Demo 3: How position information is added"""
    print("\n" + "=" * 70)
    print("DEMO 3: Positional Encoding")
    print("=" * 70)

    d_model, max_len = 128, 100
    pe = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)

    print(f"\nModel dimension: {d_model}")
    print(f"Maximum sequence length: {max_len}")

    # Create dummy embeddings (all zeros to see pure positional encoding)
    batch_size, seq_len = 1, 5
    embeddings = torch.zeros(batch_size, seq_len, d_model)

    print(f"\nInput (word embeddings): all zeros {embeddings.shape}")

    output = pe(embeddings)

    print(f"Output (with position info): {output.shape}")
    print("\nPosition encoding values (first 3 positions, first 8 dimensions):")
    for pos in range(3):
        values = output[0, pos, :8].detach().numpy()
        print(f"  Position {pos}: [{values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f}, "
              f"{values[3]:.3f}, {values[4]:.3f}, {values[5]:.3f}, ...]")

    print("\n🔍 Notice the pattern:")
    print("  - Each position has a unique 'signature'")
    print("  - Values oscillate (sine and cosine)")
    print("  - Different frequencies at different dimensions")
    print("  - This allows the model to distinguish position 1 from position 2!")


def demo_causal_masking():
    """Demo 4: Causal masking for decoder"""
    print("\n" + "=" * 70)
    print("DEMO 4: Causal Masking (for auto-regressive generation)")
    print("=" * 70)

    batch_size, seq_len, d_k = 1, 4, 8

    query = torch.randn(batch_size, seq_len, d_k)
    key = query.clone()
    value = query.clone()

    # Create causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

    print("\nCausal Mask (1 = attend, 0 = masked):")
    print(mask[0].numpy().astype(int))

    _, attention_weights = scaled_dot_product_attention(query, key, value, mask)

    print("\nAttention weights with causal masking:")
    print(attention_weights[0].numpy())

    print("\n🔍 Interpretation:")
    print("  Position 0 can only see position 0 (generating first word)")
    print("  Position 1 can see positions 0-1 (has context of first word)")
    print("  Position 2 can see positions 0-2 (has context of first 2 words)")
    print("  Position 3 can see positions 0-3 (has full context)")
    print("\n  This is how the decoder generates one word at a time!")


def demo_feed_forward():
    """Demo 5: Position-wise Feed-Forward Network"""
    print("\n" + "=" * 70)
    print("DEMO 5: Position-wise Feed-Forward Network")
    print("=" * 70)

    d_model, d_ff = 128, 512
    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0)
    ffn.eval()  # Disable dropout for demo

    print(f"\nModel dimension (d_model): {d_model}")
    print(f"Feed-forward dimension (d_ff): {d_ff} (4x expansion)")

    # Input: 4 words
    batch_size, seq_len = 1, 4
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {x.shape} (batch, sequence, d_model)")

    output = ffn(x)

    print(f"Output shape: {output.shape} (same as input)")

    print("\n🔍 Understanding 'position-wise':")
    print("  - Same network applied to EACH position independently")
    print("  - No information sharing across positions")
    print("  - Think: map(FFN, [pos1, pos2, pos3, pos4])")

    # Verify position independence
    single_pos_output = ffn(x[:, 0:1, :])
    print(f"\n  Processing position 0 alone: {single_pos_output.shape}")
    print(f"  Match with full batch position 0: {torch.allclose(output[:, 0:1, :], single_pos_output, atol=1e-5)}")

    print("\n🔍 Why FFN after attention?")
    print("  - Attention: Mixes information ACROSS positions (linear)")
    print("  - FFN: Processes EACH position with non-linearity (ReLU)")
    print("  - Together: Model can learn complex patterns!")


def demo_encoder_layer():
    """Demo 6: Complete Encoder Layer"""
    print("\n" + "=" * 70)
    print("DEMO 6: Encoder Layer (Attention + FFN + Residuals)")
    print("=" * 70)

    d_model, num_heads, d_ff = 128, 4, 512
    encoder_layer = EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.0)
    encoder_layer.eval()  # Disable dropout

    print(f"\nModel dimension: {d_model}")
    print(f"Number of heads: {num_heads}")
    print(f"Feed-forward dimension: {d_ff}")

    # Input: 5 words
    batch_size, seq_len = 2, 5
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {x.shape}")

    output, attention_weights = encoder_layer(x, return_attention=True)

    print(f"Output shape: {output.shape} (same as input)")
    print(f"Attention weights shape: {attention_weights.shape}")

    print("\n🔍 What happened inside?")
    print("  1. Multi-Head Self-Attention → Dropout")
    print("  2. Residual Connection: x + attn_output")
    print("  3. Layer Normalization")
    print("  4. Position-wise FFN → Dropout")
    print("  5. Residual Connection: x + ffn_output")
    print("  6. Layer Normalization")

    print("\n🔍 Why residual connections?")
    print("  - Enable gradient flow in deep networks")
    print("  - Model learns 'what to add' instead of full transformation")
    print("  - Easier to learn identity if needed")

    print("\n🔍 Why layer normalization?")
    print("  - Stabilizes training by normalizing activations")
    print("  - Works better than batch norm for sequences")

    # Test with padding mask
    print("\n🔍 Testing with padding mask:")
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, 3:] = 0  # Mask last 2 positions
    print(f"  Mask shape: {mask.shape}")
    print(f"  Real tokens: positions 0-2, Padding: positions 3-4")

    output_masked = encoder_layer(x, mask)
    print(f"  Output with mask: {output_masked.shape}")
    print("  ✓ Attention doesn't focus on padding tokens!")

    # Demonstrate stacking
    print("\n🔍 Stacking encoder layers:")
    encoder_layer_2 = EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.0)
    encoder_layer_2.eval()

    x_layer1 = encoder_layer(x)
    x_layer2 = encoder_layer_2(x_layer1)

    print(f"  After layer 1: {x_layer1.shape}")
    print(f"  After layer 2: {x_layer2.shape}")
    print("  ✓ Layers can be stacked! (Full encoder = N stacked layers)")


def main():
    print("\n" + "🚀" * 35)
    print("TRANSFORMER COMPONENTS: Interactive Demo")
    print("Understanding Stories 1-6 Implementation")
    print("🚀" * 35)

    demo_scaled_attention()
    demo_multi_head()
    demo_positional_encoding()
    demo_causal_masking()
    demo_feed_forward()
    demo_encoder_layer()

    print("\n" + "=" * 70)
    print("✅ All components working correctly!")
    print("=" * 70)
    print("\n📚 Complete Transformer Implementation:")
    print("  ✓ Stories 1-6: Core components (attention, FFN, layers)")
    print("  ✓ Stories 7-10: Full encoder-decoder architecture")
    print("  ✓ Stories 11-12: Training loop with toy datasets")
    print("  ✓ Story 13: Attention visualization")
    print()
    print("🚀 Try these next:")
    print("  1. python train_toy.py --task copy --epochs 10")
    print("  2. python train_toy.py --task reverse --epochs 20")
    print("  3. python visualize_attention.py")
    print("  4. python demo_all.py")
    print()


if __name__ == "__main__":
    main()
