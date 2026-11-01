#!/usr/bin/env python3
"""Testing and Inference Script for JAX GPT-2.

Features:
- Load model weights directly from a file or checkpoint path via `--weights_path`.
- Autoregressive text generation with temperature and top-k sampling.
- Interactive generation shell via `--interactive`.
- Quantitative evaluation: compute validation loss and perplexity on binary datasets.
- Inspect training data batches and tokens via `--inspect_data`.
"""

import argparse
import math
from pathlib import Path
import sys
from typing import Optional

import jax
import jax.numpy as jnp
import tiktoken

from dataset import DataLoader
from gpt import GPT2, GPTConfig, count_parameters, gpt2_loss, init_gpt2
from utils import CheckpointManager, load_checkpoint, load_model_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test, evaluate, or generate text from JAX GPT-2 checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Weights and model configuration
    parser.add_argument(
        "--weights_path",
        type=Path,
        default=None,
        help="Path to checkpoint file (.ckpt or .pkl). If omitted, auto-detects best/latest checkpoint in checkpoint_dir.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("artifacts/checkpoints"),
        help="Directory to search for checkpoints if --weights_path is omitted",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["53m", "68m"],
        default="53m",
        help="Fallback model variant if checkpoint has no embedded config: '53m' or '68m'",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=768,
        help="Sequence context length",
    )

    # Text Generation parameters
    parser.add_argument(
        "--prompt",
        type=str,
        default="The Roman goddess of crafts and wisdom",
        help="Initial text prompt for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=60,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0.0 = greedy argmax, >0.0 = stochastic sampling)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-K sampling filter (0 or negative to disable)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed for generation sampling",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal prompt session",
    )

    # Quantitative Evaluation
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate model loss and perplexity on dataset",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("artifacts/data/val.bin"),
        help="Dataset path (.bin) for evaluation or inspection",
    )
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=50,
        help="Number of batches to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )

    # Dataset Inspection
    parser.add_argument(
        "--inspect_data",
        action="store_true",
        help="Inspect first sequence in dataset (token IDs and decoded text)",
    )

    return parser.parse_args()


def resolve_weights(weights_path: Optional[Path], checkpoint_dir: Path) -> Optional[Path]:
    """Finds target checkpoint path from arguments or directory search."""
    if weights_path is not None:
        p = Path(weights_path)
        if not p.is_file():
            raise FileNotFoundError(f"Specified weights file not found: {p}")
        return p.resolve()

    # Auto-detect best or latest checkpoint in checkpoint_dir
    if checkpoint_dir.is_dir():
        best = checkpoint_dir / "best_model.ckpt"
        if best.is_file():
            return best.resolve()

        manager = CheckpointManager(directory=checkpoint_dir)
        latest = manager.latest_checkpoint()
        if latest is not None:
            return latest.resolve()

    return None


def inspect_first_datapoint(data_path: Path, seq_len: int, tokenizer: tiktoken.Encoding):
    """Loads and prints the first data sequence encoded and decoded."""
    print(f"\n--- Inspecting First Batch from {data_path} ---")
    if not data_path.is_file():
        print(f"Data file not found: {data_path}")
        return

    try:
        loader = DataLoader(
            str(data_path),
            batch_size=1,
            seq_len=seq_len,
            shuffle=False,
            return_jax=False,
        )
    except ValueError as e:
        print(f"Cannot inspect sequence of length {seq_len}: {e}")
        return
    inputs, targets = loader.next_batch()
    input_ids = inputs[0].tolist()
    target_ids = targets[0].tolist()

    print(f"First {min(20, len(input_ids))} Input Token IDs:")
    print(f"  {input_ids[:20]} ... (total {len(input_ids)} tokens)")

    print("\nDecoded Input Text:")
    print("-" * 50)
    print(tokenizer.decode(input_ids))
    print("-" * 50)

    print("\nDecoded Target Text (input shifted by 1):")
    print("-" * 50)
    print(tokenizer.decode(target_ids))
    print("-" * 50)


def evaluate_dataset(
    model_params: dict,
    config: GPTConfig,
    data_path: Path,
    batch_size: int,
    seq_len: int,
    max_batches: int,
) -> tuple[float, float]:
    """Computes cross-entropy loss and perplexity on a binary token file."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {data_path}")

    try:
        loader = DataLoader(
            str(data_path),
            batch_size=batch_size,
            seq_len=seq_len,
            shuffle=False,
            return_jax=True,
        )
    except ValueError as e:
        print(f"\n[Notice] Cannot evaluate on {data_path}: {e}")
        return float("nan"), float("nan")

    @jax.jit
    def step_eval(p, x, y):
        loss, _ = gpt2_loss(p, x, y, config, deterministic=True)
        return loss

    total_loss = 0.0
    num_batches = 0

    print(f"\nEvaluating on {data_path} (batch_size={batch_size}, seq_len={seq_len})...")
    for _ in range(max_batches):
        try:
            x, y = loader.next_batch()
            l = float(step_eval(model_params, x, y))
            total_loss += l
            num_batches += 1
        except StopIteration:
            break

    mean_loss = total_loss / max(1, num_batches)
    perplexity = math.exp(min(20.0, mean_loss))
    return mean_loss, perplexity


def generate_text(
    model: GPT2,
    params: dict,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    rng: jax.Array,
) -> str:
    """Encodes prompt, runs autoregressive generation, and decodes back to string."""
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.eot_token]

    prompt_arr = jnp.array([prompt_ids], dtype=jnp.int32)
    top_k_val = top_k if top_k > 0 else None

    gen_arr = model.generate(
        params=params,
        prompt_tokens=prompt_arr,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k_val,
        rng=rng,
    )
    return tokenizer.decode(gen_arr[0].tolist())


def main():
    args = parse_args()
    tokenizer = tiktoken.get_encoding("gpt2")

    # Optional inspection mode
    if args.inspect_data:
        inspect_first_datapoint(args.data_path, args.seq_len, tokenizer)
        if not args.eval and args.weights_path is None:
            return

    # 1. Resolve weights file
    target_path = resolve_weights(args.weights_path, args.checkpoint_dir)
    params = None
    config = None
    loaded_step = None
    loaded_metrics = {}

    if target_path is not None:
        print(f"\n[Loading Weights] from: {target_path}")
        ckpt = load_checkpoint(target_path, restore_to_device=True)
        params = ckpt["model_params"]
        config = ckpt.get("config")
        loaded_step = ckpt.get("step")
        loaded_metrics = ckpt.get("metrics", {})
        print(f"  Step:    {loaded_step}")
        if loaded_metrics:
            print(f"  Metrics: {loaded_metrics}")
    else:
        print("\n[Notice] No checkpoint found or specified via --weights_path.")
        print("  Initializing fresh model with random weights for demonstration.")

    # 2. Model Configuration
    if config is None:
        if args.variant == "53m":
            config = GPTConfig.gpt2_53m(max_seq_len=args.seq_len)
        else:
            config = GPTConfig.gpt2_68m(max_seq_len=args.seq_len)

    model = GPT2(config)

    # If no checkpoint was loaded, initialize weights now
    if params is None:
        init_rng = jax.random.PRNGKey(args.seed)
        params = model.init(init_rng)

    n_params = count_parameters(params)
    print("\nModel Architecture:")
    print(f"  Variant:          {config.n_layers} layers, {config.d_model} d_model, {config.n_heads} heads")
    print(f"  Context Window:   {config.max_seq_len} tokens")
    print(f"  Total Parameters: {n_params:,} (~{n_params / 1e6:.2f}M)")

    # 3. Quantitative Evaluation (if requested)
    if args.eval:
        eval_path = args.data_path
        loss, ppl = evaluate_dataset(
            model_params=params,
            config=config,
            data_path=eval_path,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            max_batches=args.eval_batches,
        )
        print("-" * 50)
        print(f"Evaluation Results on {eval_path.name}:")
        print(f"  Loss:       {loss:.4f}")
        print(f"  Perplexity: {ppl:.2f}")
        print("-" * 50)

    # 4. Text Generation
    rng = jax.random.PRNGKey(args.seed)

    if args.interactive:
        print("\n" + "=" * 60)
        print("Entering Interactive Generation Shell (type 'exit' or 'quit' to stop)")
        print(f"Sampling: temperature={args.temperature}, top_k={args.top_k}")
        print("=" * 60)
        while True:
            try:
                user_prompt = input("\nPrompt > ").strip()
                if not user_prompt:
                    continue
                if user_prompt.lower() in ("exit", "quit", "q"):
                    break
                rng, k_gen = jax.random.split(rng)
                output = generate_text(
                    model=model,
                    params=params,
                    tokenizer=tokenizer,
                    prompt=user_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    rng=k_gen,
                )
                print("\nGenerated:")
                print(output)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive shell.")
                break
    else:
        rng, k_gen = jax.random.split(rng)
        print(f"\n--- Generating text from prompt ---")
        print(f"Prompt: \"{args.prompt}\"")
        output = generate_text(
            model=model,
            params=params,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            rng=k_gen,
        )
        print("\nOutput:")
        print("-" * 60)
        print(output)
        print("-" * 60)


if __name__ == "__main__":
    main()

