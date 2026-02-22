# Transformer: Educational PyTorch Implementation

A toy implementation of the Transformer architecture from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017).

## 🎯 Learning Objectives

This project is designed for **educational purposes** to understand the Transformer architecture through implementation. The focus is on:

- **Clarity over Performance**: Code is written to mirror the paper's equations explicitly
- **Small Scale**: Toy model dimensions (d_model=128, h=4 heads) for fast CPU training
- **Comprehensive Comments**: Extensive documentation explaining the "why" behind each component
- **Progressive Learning**: Build complexity incrementally from attention mechanisms to full model

## 📚 What You'll Learn

1. **Scaled Dot-Product Attention**: The fundamental attention mechanism
2. **Multi-Head Attention**: Parallel attention heads for diverse representations
3. **Positional Encoding**: Injecting sequence order information
4. **Encoder Architecture**: Self-attention and feed-forward layers
5. **Decoder Architecture**: Masked attention and cross-attention
6. **Training**: Adam optimizer, learning rate scheduling, label smoothing
7. **Validation**: Simple tasks (copy, reverse) to verify implementation

## 🚀 Quick Start

### Installation

```bash
# Using uv package manager
uv sync

# Or manually with pip
pip install -r requirements.txt
```

### Project Structure

```
transformer-attention/
├── transformer/          # Main implementation
│   ├── attention.py     # Scaled dot-product & multi-head attention
│   ├── layers.py        # Encoder & decoder layers
│   ├── model.py         # Complete Transformer model
│   ├── positional.py    # Positional encodings
│   └── utils.py         # Helper functions
├── tests/               # Unit tests
├── notebooks/           # Jupyter notebooks for exploration
├── pyproject.toml       # Project configuration
└── README.md           # This file
```

### Running Tests

```bash
pytest tests/
```

## 📖 Implementation Roadmap

This project follows a structured implementation plan across 13 stories:

### Foundation (Stories 1-6)
- ✅ Story 01: Project Setup
- ✅ Story 02: Scaled Dot-Product Attention
- ✅ Story 03: Multi-Head Attention
- ✅ Story 04: Positional Encoding
- ⏳ Story 05: Feed-Forward Network
- ⏳ Story 06: Encoder Layer

### Architecture Assembly (Stories 7-10)
- ⏳ Story 07: Encoder Stack
- ⏳ Story 08: Decoder Layer
- ⏳ Story 09: Decoder Stack
- ⏳ Story 10: Full Transformer Model

### Training & Validation (Stories 11-13)
- ⏳ Story 11: Training Loop
- ⏳ Story 12: Toy Datasets (Copy/Reverse Tasks)
- ⏳ Story 13: Attention Visualization

## 🎓 Key Differences from Production Models

| Aspect | This Implementation | Paper/Production |
|--------|-------------------|------------------|
| d_model | 128 | 512 |
| num_heads | 4 | 8 |
| num_layers | 2 | 6 |
| d_ff | 512 | 2048 |
| vocab_size | 1000 | 30,000+ |
| Training | Toy tasks (copy/reverse) | WMT translation |
| Hardware | CPU | Multi-GPU |

## 📝 References

- **Paper**: [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- **Annotated Transformer**: [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- **PyTorch Documentation**: [torch.nn](https://pytorch.org/docs/stable/nn.html)

## 🤝 Learning Philosophy

This implementation prioritizes:
1. **Understanding**: Extensive comments explaining mathematical intuitions
2. **Testability**: Each component has unit tests with shape validation
3. **Modularity**: Clear separation of concerns for easy debugging
4. **Reproducibility**: Fixed random seeds and deterministic operations

## 📊 Expected Results

When trained on toy tasks:
- **Copy Task**: >99% accuracy within 5 epochs
- **Reverse Task**: >95% accuracy within 10 epochs

If these benchmarks aren't met, the implementation needs debugging!

## 🔧 Development

```bash
# Run tests
pytest tests/ -v

# Run specific test file
pytest tests/test_attention.py

# Generate coverage report
pytest --cov=transformer tests/
```

## 📄 License

Educational project - use freely for learning purposes.

## 🙏 Acknowledgments

Based on the groundbreaking work by Vaswani et al. in "Attention Is All You Need".
