#!/usr/bin/env python3
"""Pretraining Script for JAX GPT-2 (53M & 68M variants).

Features:
- Pure JAX training loop with `jax.jit` and `jax.value_and_grad`.
- Decoupled weight decay via pure JAX AdamW with 1D bias/norm parameter filtering.
- Variable learning rate scheduling with linear warmup and cosine decay.
- Global gradient norm clipping for training stability.
- Zero-copy memory-mapped token streaming via JAX DataLoader.
- Periodic evaluation, validation perplexity calculation, and sample text generation.
- Automated checkpointing with `CheckpointManager` (pruning + tracking best_model.ckpt).
- Seamless training resumption via `--resume`.
"""

import argparse
from datetime import datetime
import math
from pathlib import Path
import time
from typing import Optional

import jax
import jax.numpy as jnp
import tiktoken

from dataset import DataLoader
from gpt import GPT2, GPTConfig, count_parameters, gpt2_loss, init_gpt2
from utils import (
    AdamW,
    CheckpointManager,
    cosine_schedule_with_warmup,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain JAX GPT-2 on tokenized binary corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model architecture
    parser.add_argument(
        "--variant",
        type=str,
        choices=["53m", "68m"],
        default="53m",
        help="Target GPT-2 model size: '53m' (L=2, d=768) or '68m' (L=4, d=768)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=768,
        help="Context sequence length",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability (0.0 recommended for pretraining)",
    )

    # Data paths
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("artifacts/data"),
        help="Directory containing train.bin and val.bin",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("artifacts/checkpoints"),
        help="Directory to save training checkpoints",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Per-step batch size",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5000,
        help="Total training optimization steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=6e-4,
        help="Peak learning rate for AdamW",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=6e-5,
        help="Minimum floor learning rate after cosine decay",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of steps for linear learning rate warmup",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Decoupled weight decay coefficient",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for global L2 clipping",
    )

    # Logging and evaluation
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Steps between console training loss logs",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=250,
        help="Steps between validation evaluations",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=25,
        help="Number of batches to evaluate on validation set",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=500,
        help="Steps between autoregressive sample generation previews",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=500,
        help="Steps between saving regular checkpoints",
    )
    parser.add_argument(
        "--max_to_keep",
        type=int,
        default=3,
        help="Maximum number of step checkpoints to retain",
    )

    # Resumption & seed
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from, or 'latest' to auto-resume",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weight initialization and data shuffling",
    )

    return parser.parse_args()


def get_model_config(variant: str, seq_len: int, dropout: float) -> GPTConfig:
    """Returns the requested GPTConfig."""
    if variant == "53m":
        cfg = GPTConfig.gpt2_53m(max_seq_len=seq_len)
    elif variant == "68m":
        cfg = GPTConfig.gpt2_68m(max_seq_len=seq_len)
    else:
        raise ValueError(f"Unknown variant '{variant}'. Choose '53m' or '68m'.")

    if dropout != cfg.dropout_rate:
        cfg = GPTConfig(
            vocab_size=cfg.vocab_size,
            max_seq_len=cfg.max_seq_len,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            bias=cfg.bias,
            tie_word_embeddings=cfg.tie_word_embeddings,
            dropout_rate=dropout,
        )
    return cfg


def evaluate(
    val_loader: DataLoader,
    params: dict,
    config: GPTConfig,
    eval_steps: int,
) -> tuple[float, float]:
    """Computes mean validation loss and perplexity across eval_steps batches."""
    @jax.jit
    def eval_batch(p, x, y):
        l, _ = gpt2_loss(p, x, y, config, deterministic=True)
        return l

    total_loss = 0.0
    val_loader.reset()

    for _ in range(eval_steps):
        try:
            x, y = val_loader.next_batch()
            l = float(eval_batch(params, x, y))
            total_loss += l
        except StopIteration:
            break

    mean_loss = total_loss / max(1, eval_steps)
    perplexity = math.exp(min(20.0, mean_loss))
    return mean_loss, perplexity


def sample_generation(
    model: GPT2,
    params: dict,
    tokenizer: tiktoken.Encoding,
    prompt: str = "In a faraway kingdom",
    max_new_tokens: int = 30,
    rng: Optional[jax.Array] = None,
) -> str:
    """Generates sample text for monitoring training progression."""
    prompt_ids = tokenizer.encode(prompt)
    prompt_arr = jnp.array([prompt_ids], dtype=jnp.int32)
    gen_tokens = model.generate(
        params=params,
        prompt_tokens=prompt_arr,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=40,
        rng=rng,
    )
    return tokenizer.decode(gen_tokens[0].tolist())


def main():
    args = parse_args()

    print("=" * 70)
    print(f"JAX GPT-2 Pretraining: {args.variant.upper()} Variant")
    print(f"  Device:           {jax.devices()[0]}")
    print(f"  Sequence Length:  {args.seq_len}")
    print(f"  Batch Size:       {args.batch_size} (Tokens/step: {args.batch_size * args.seq_len:,})")
    print(f"  Max Steps:        {args.max_steps:,}")
    print(f"  Peak LR:          {args.learning_rate:g} (min: {args.min_lr:g}, warmup: {args.warmup_steps})")
    print("=" * 70)

    # 1. Verify Data Files
    train_bin = args.data_dir / "train.bin"
    val_bin = args.data_dir / "val.bin"

    if not train_bin.is_file():
        raise FileNotFoundError(
            f"Training file not found: {train_bin}. "
            "Please run 'python prepare_dataset.py' first."
        )

    # 2. DataLoaders
    try:
        train_loader = DataLoader(
            train_bin,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            shuffle=True,
            seed=args.seed,
            return_jax=True,
        )
    except ValueError as e:
        raise ValueError(
            f"{e}\n"
            f"The training file '{train_bin}' is too small for seq_len={args.seq_len}.\n"
            "Please run 'python prepare_dataset.py' to generate your pretraining corpus, "
            "or reduce --seq_len for testing."
        ) from e

    val_loader = None
    if val_bin.is_file():
        try:
            val_loader = DataLoader(
                val_bin,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                shuffle=False,
                return_jax=True,
            )
        except ValueError as e:
            print(f"\n[Notice] Validation disabled: {e}")
            print(f"  '{val_bin}' only has a tiny sample. Run 'prepare_dataset.py' to generate full validation data.\n")
            val_loader = None

    # 3. Model Configuration & Tokenizer
    config = get_model_config(args.variant, args.seq_len, args.dropout)
    model = GPT2(config)
    tokenizer = tiktoken.get_encoding("gpt2")

    # 4. Checkpoint Manager & Resumption
    checkpoint_manager = CheckpointManager(
        directory=args.checkpoint_dir,
        max_to_keep=args.max_to_keep,
        metric_name="val_loss",
        mode="min",
    )

    rng = jax.random.PRNGKey(args.seed)
    rng_init, rng_train = jax.random.split(rng)

    start_step = 0
    params = None
    opt_state = None

    # Handle resume flag
    resume_path = None
    if args.resume:
        if args.resume.lower() == "latest":
            resume_path = checkpoint_manager.latest_checkpoint()
        else:
            resume_path = Path(args.resume)

    if resume_path and resume_path.is_file():
        print(f"\nResuming training from checkpoint: {resume_path}")
        ckpt_data = load_checkpoint(resume_path, restore_to_device=True, dataloader=train_loader)
        params = ckpt_data["model_params"]
        opt_state = ckpt_data["opt_state"]
        start_step = ckpt_data["step"]
        if ckpt_data["config"]:
            config = ckpt_data["config"]
            model = GPT2(config)
        print(f"  Resumed at step {start_step:,} (metrics: {ckpt_data.get('metrics', {})})")
    else:
        print("\nInitializing fresh model weights...")
        params = model.init(rng_init)

    total_params = count_parameters(params)
    print(f"  Total Parameters: {total_params:,} (~{total_params / 1e6:.2f}M)")

    # 5. Optimizer & Variable LR Scheduler
    lr_schedule = cosine_schedule_with_warmup(
        peak_lr=args.learning_rate,
        warmup_steps=args.warmup_steps,
        total_steps=args.max_steps,
        min_lr=args.min_lr,
    )

    optimizer = AdamW(
        learning_rate=lr_schedule,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
    )

    if opt_state is None:
        opt_state = optimizer.init(params)

    # 6. JIT Training Step
    tokens_per_step = args.batch_size * args.seq_len

    @jax.jit
    def train_step(p, s, x, y, k_step):
        def compute_loss(params_inner):
            l, _ = gpt2_loss(
                params_inner,
                x,
                y,
                config,
                deterministic=(args.dropout == 0.0),
                rng=k_step,
            )
            return l

        loss_val, grads = jax.value_and_grad(compute_loss)(p)
        new_p, new_s, metrics = optimizer.step(p, grads, s)
        metrics["loss"] = loss_val
        return new_p, new_s, metrics

    print("\nStarting Pretraining Loop...")
    print("-" * 70)

    t_start = time.time()
    t_prev = t_start
    best_val_loss = float("inf")

    try:
        for step in range(start_step + 1, args.max_steps + 1):
            t0 = time.time()

            # Fetch batch
            inputs, targets = train_loader.next_batch()

            # Step key for stochastic dropout (if enabled)
            rng_train, k_step = jax.random.split(rng_train)

            # Optimization step
            params, opt_state, metrics = train_step(params, opt_state, inputs, targets, k_step)

            loss = float(metrics["loss"])
            curr_lr = float(metrics["lr"])
            grad_norm = float(metrics["grad_norm"])

            t1 = time.time()
            dt = t1 - t0
            tok_per_sec = tokens_per_step / max(1e-6, dt)

            # 7. Logging
            if step % args.log_interval == 0 or step == 1:
                elapsed = time.time() - t_start
                print(
                    f"step {step:6d}/{args.max_steps} | "
                    f"loss: {loss:6.4f} | "
                    f"lr: {curr_lr:.2e} | "
                    f"grad: {grad_norm:5.2f} | "
                    f"{tok_per_sec:7,.0f} tok/s | "
                    f"{dt * 1000:5.1f}ms/step"
                )

            # 8. Evaluation
            val_loss = None
            val_ppl = None
            if val_loader is not None and (step % args.eval_interval == 0 or step == args.max_steps):
                print(f"\n--- Evaluating Validation Split at Step {step} ---")
                val_loss, val_ppl = evaluate(val_loader, params, config, args.eval_steps)
                print(f"  val_loss: {val_loss:.4f} | val_ppl: {val_ppl:.2f}\n")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            # 9. Sample Text Generation Preview
            if step % args.sample_interval == 0 or step == 1:
                rng_train, k_gen = jax.random.split(rng_train)
                prompt = "The Roman goddess"
                preview = sample_generation(model, params, tokenizer, prompt=prompt, max_new_tokens=25, rng=k_gen)
                print(f"\n[Sample Step {step}] '{prompt}...' ->\n  \"{preview}\"\n")

            # 10. Checkpointing
            if step % args.save_interval == 0 or step == args.max_steps:
                save_metrics = {
                    "train_loss": loss,
                    "val_loss": val_loss if val_loss is not None else loss,
                    "lr": curr_lr,
                    "step": step,
                }
                saved_ckpt = checkpoint_manager.save(
                    step=step,
                    model_params=params,
                    opt_state=opt_state,
                    config=config,
                    metrics=save_metrics,
                    dataloader=train_loader,
                )
                print(f"[Checkpoint Saved] -> {saved_ckpt.name}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving emergency checkpoint...")
        emergency_path = checkpoint_manager.save(
            step=step,
            model_params=params,
            opt_state=opt_state,
            config=config,
            metrics={"train_loss": loss, "interrupted": True},
            dataloader=train_loader,
        )
        print(f"Emergency checkpoint saved to: {emergency_path}")

    total_time = time.time() - t_start
    print("=" * 70)
    print(f"Pretraining Finished in {total_time:.1f}s ({total_time / 60:.2f} min)")
    print(f"Best Checkpoint: {checkpoint_manager.best_checkpoint()}")
    print("=" * 70)


if __name__ == "__main__":
    main()

