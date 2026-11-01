# JaxGPT

Pure **JAX** implementation of GPT-2 pretraining optimized for small language models (**53M** and **68M** parameter variants) on **Apple Silicon (Metal/MPS)** and **NVIDIA GPUs**.

Built from first principles without external high-level frameworks (no Flax, Haiku, Equinox, or Optax)—featuring pure JAX layers, custom fused self-attention, AdamW optimizer with decoupled weight decay, learning rate schedules, zero-copy memory-mapped data loading, and atomic checkpointing.

---

## Table of Contents
- [Architecture & Model Variants](#architecture--model-variants)
- [Installation & Environment](#installation--environment)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Testing & Inference](#testing--inference)
- [Hardware Acceleration (MPS / CUDA / CPU)](#hardware-acceleration)

---

## Architecture & Model Variants

| Variant | Layers ($L$) | Hidden Size ($d_{\text{model}}$) | Heads ($h$) | Context Length | Total Parameters | Active Weights (excl. embeddings) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`53m`** | 2 | 768 | 12 | 768 | 53,561,088 (~53.5M) | ~14.7M |
| **`68m`** | 4 | 768 | 12 | 768 | 67,736,832 (~67.7M) | ~28.9M |

- **Weight Tying**: Output projection matrix shares weights with token embeddings (`wte.T`).
- **Learned Positional Embeddings**: Standard GPT-2 absolute positional embeddings (`wpe`).
- **Pre-LayerNorm Architecture**: LayerNorm applied before attention and MLP blocks with float32 upcasting for stability.
- **Residual Scaling**: Residual projection weights scaled by $1 / \sqrt{2L}$ to prevent variance explosion in deep stacks.

---

## Installation & Environment

This repository uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

```bash
# Clone the repository
git clone https://github.com/your-username/jaxgpt.git
cd jaxgpt

# Install all dependencies into a virtual environment
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

---

## Dataset Preparation

Prepares a balanced pretraining corpus tokenized with the GPT-2 tokenizer (`tiktoken`, vocab 50,257):
- **50% FineWeb-Edu** (`HuggingFaceFW/fineweb-edu: sample-10BT`): High-quality educational text.
- **30% Simple English Wikipedia** (`wikimedia/wikipedia: 20231101.simple`): Core encyclopedic knowledge.
- **20% TinyStories** (`roneneldan/TinyStories`): Narrative grammar, story coherence, and syntax.

The script uses a **Download-First architecture** with smart sharding: only downloading the required parquet shards locally before fast multi-threaded packing with PyArrow (immune to HTTP read timeouts).

### Option A: Set Hugging Face Token (Recommended)
```bash
export HF_TOKEN="your_hf_token_here"
```

### Option B: Full Pretraining Corpus (500M tokens)
```bash
python prepare_dataset.py \
  --output_dir artifacts/data \
  --total_tokens 500000000 \
  --val_tokens 5000000
```

### Option C: Quick Test Dataset (e.g. 50k tokens)
```bash
python prepare_dataset.py \
  --output_dir artifacts/data \
  --total_tokens 50000 \
  --val_tokens 5000
```

Outputs packed uint16 binary files: `artifacts/data/train.bin`, `artifacts/data/val.bin`, and `meta.json`.

---

## Training

Run pretraining using `train.py`. The training loop features `@jax.jit` compiled optimization steps, global gradient clipping, periodic validation, and atomic checkpointing (`best_model.ckpt`).

### 1. Pretrain the 53M Model
```bash
python train.py \
  --variant 53m \
  --batch_size 16 \
  --seq_len 768 \
  --max_steps 5000 \
  --learning_rate 6e-4 \
  --warmup_steps 500 \
  --eval_interval 250 \
  --save_interval 500 \
  --sample_interval 500
```

### 2. Pretrain the 68M Model (Recommended for Better Coherence)
The 68M variant doubles the layer depth to 4 layers ($L=4, d=768, h=12$), providing significantly better entity separation:
```bash
python train.py \
  --variant 68m \
  --batch_size 16 \
  --seq_len 768 \
  --max_steps 10000 \
  --learning_rate 6e-4 \
  --warmup_steps 1000 \
  --eval_interval 250 \
  --save_interval 500
```

### 3. Quick Smoke Test (50 Steps)
To quickly test that dataloading, JIT compilation, and checkpointing run smoothly:
```bash
python train.py \
  --variant 53m \
  --batch_size 8 \
  --max_steps 50 \
  --log_interval 5 \
  --eval_interval 25 \
  --save_interval 25 \
  --sample_interval 25
```

### 4. Resume Training from Checkpoint
Automatically detects and resumes from the latest checkpoint in `artifacts/checkpoints/`:
```bash
python train.py --variant 53m --resume --max_steps 15000
```

---

## Testing & Inference

Use `test.py` to generate text, run an interactive shell, or evaluate validation metrics against saved checkpoints.

### 1. Generate Text from a Prompt
```bash
python test.py \
  --weights_path artifacts/checkpoints/best_model.ckpt \
  --prompt "Once upon a time" \
  --max_new_tokens 80 \
  --temperature 0.75 \
  --top_k 30
```

### 2. Interactive Generation Shell
Launches an interactive prompt session:
```bash
python test.py \
  --weights_path artifacts/checkpoints/best_model.ckpt \
  --interactive
```

### 3. Evaluate Validation Perplexity
Computes cross-entropy loss and perplexity on the binary validation dataset:
```bash
python test.py \
  --weights_path artifacts/checkpoints/best_model.ckpt \
  --eval \
  --eval_batches 50
```

### 4. Inspect Training Batches
Inspect raw tokens and decoded text directly from the binary dataset:
```bash
python test.py --inspect_data --inspect_batches 2
```

---

## Hardware Acceleration

JAX automatically detects and leverages your hardware device:

### Apple Silicon (Metal / MPS)
On macOS (M1/M2/M3/M4), JAX uses `jax-metal` to run matrix multiplications and attention directly on the unified GPU.
```bash
# Verify Metal acceleration
python -c "import jax; print('JAX Devices:', jax.devices())"
# Output: [METAL(id=0)] or [MPS:0]
```

### NVIDIA GPU (CUDA)
To run on Linux with NVIDIA GPUs, install the CUDA-enabled JAX wheel:
```bash
pip install --upgrade "jax[cuda12]"
```

### CPU Fallback
To force CPU execution (e.g. in headless CI environments):
```bash
export JAX_PLATFORMS=cpu
python train.py --variant 53m --max_steps 100
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
