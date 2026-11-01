# JaxGPT

Pretraining GPT-2 (50M–70M parameter variants) in JAX and Apple Silicon Metal (MPS).

## Project Structure

```
jaxgpt/
├── dataset/
│   ├── __init__.py       # Exports DataLoader
│   └── dataloader.py     # Zero-copy, memory-mapped JAX DataLoader
├── gpt/
│   ├── __init__.py
│   └── gpt.py            # GPT model implementation
├── artifacts/            # (Git-ignored) Datasets, checkpoints, training logs
│   └── data/             # train.bin, val.bin, meta.json
├── prepare_dataset.py    # Script to download, mix (50/30/20), and tokenize corpus
├── pyproject.toml
└── uv.lock
```

## Dataset Mixture (500M Tokens)

Targeting a balanced curriculum for small language models:
- **50% FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`, `sample-10BT`): High-quality educational text, factual explanations, scientific concepts (~250M tokens).
- **30% Simple English Wikipedia** (`wikimedia/wikipedia`, `20231101.simple`): Core historical, geographical, and encyclopedic facts (~150M tokens via multi-epoch shuffled cycling).
- **20% TinyStories** (`roneneldan/TinyStories`): Narrative coherence, character tracking, and syntax grounding (~100M tokens).

Tokenized using GPT-2 BPE (`tiktoken`, vocab size 50,257, `<|endoftext|>` = 50256) into compact `uint16` binary files (`train.bin`, `val.bin`).

## Quickstart

### 1. Environment Setup (uv)
```bash
# Sync dependencies
uv sync
```

### 2. Prepare Dataset
```bash
# Full 500M token pretraining corpus + 5M validation (saved to artifacts/data/)
python prepare_dataset.py --output_dir artifacts/data --total_tokens 500000000 --val_tokens 5000000

# Quick test run (e.g. 50k tokens)
python prepare_dataset.py --output_dir artifacts/data --total_tokens 50000 --val_tokens 5000
```

### 3. Using the JAX DataLoader
```python
from dataset import DataLoader

# Training loader (random sampling across corpus, zero RAM overhead via mmap)
train_loader = DataLoader(
    "artifacts/data/train.bin",
    batch_size=32,
    seq_len=512,
    shuffle=True,
    return_jax=True,  # Directly produces jax.Array on device
)

# Validation loader (deterministic sequential evaluation)
val_loader = DataLoader(
    "artifacts/data/val.bin",
    batch_size=32,
    seq_len=512,
    shuffle=False,
    return_jax=True,
)

# Fetch a batch: targets are shifted by 1 token relative to inputs
inputs, targets = train_loader.next_batch()
# inputs:  (32, 512) int32 jax.Array
# targets: (32, 512) int32 jax.Array
```
