#!/usr/bin/env python3
"""Dataset Preparation Script for JAX GPT-2 (50M-70M parameters).

Prepares a balanced pretraining corpus split:
  - 50% FineWeb-Edu (HuggingFaceFW/fineweb-edu: sample-10BT)
  - 30% Simple English Wikipedia (wikimedia/wikipedia: 20231101.simple)
  - 20% TinyStories (roneneldan/TinyStories)

Download Strategy:
  - Always downloads required parquet/arrow data locally before tokenizing.
  - No live HTTP streaming during tokenization: immune to network timeouts and connection drops.
  - Downloads only the necessary shards for FineWeb-Edu (~800 MB) and TinyStories (~520 MB)
    rather than downloading entire multi-gigabyte/terabyte subsets.
  - Once downloaded, subsequent runs load from local disk in seconds.

Outputs packed uint16 binary files (.bin) into artifacts/data/ ready for zero-copy JAX DataLoader memory-mapping.
"""

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple
import numpy as np
import pyarrow.parquet as pq
import tiktoken
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


# ==============================================================================
# Authentication & Network Helpers
# ==============================================================================

def resolve_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """Resolves Hugging Face authentication token from CLI or environment."""
    if cli_token and cli_token.strip():
        return cli_token.strip()
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()
    return None


def download_hf_file(
    repo_id: str,
    filename: str,
    local_dir: Path,
    token: Optional[str] = None,
    max_retries: int = 3,
) -> Path:
    """Downloads a file from Hugging Face Hub with automatic resume and retry."""
    target_path = local_dir / filename
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    for attempt in range(max_retries):
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(local_dir),
                token=token,
            )
            return Path(downloaded)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_sec = 2 ** (attempt + 1)
                print(f"[Retry {attempt + 1}/{max_retries}] Download {filename} failed: {e}. Retrying in {wait_sec}s...")
                time.sleep(wait_sec)
            else:
                print(f"[Error] Failed to download {filename} from {repo_id}: {e}")
                raise


# ==============================================================================
# Dataset Downloaders
# ==============================================================================

def download_tinystories(
    raw_dir: Path,
    token: Optional[str] = None,
    needed_train_tokens: int = 100_000_000,
) -> Tuple[List[Path], List[Path]]:
    """Downloads TinyStories validation and required training parquet files."""
    dest_dir = raw_dir / "tinystories"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check for already downloaded local files
    existing_val = sorted(dest_dir.glob("*validation*.parquet"))
    existing_train = sorted(dest_dir.glob("*train*.parquet"))
    if existing_val and existing_train:
        print(f"  * [TinyStories] Using local cache ({len(existing_val)} val, {len(existing_train)} train files)")
        return existing_val, existing_train

    print("  * [TinyStories] Downloading parquet files from roneneldan/TinyStories...")
    api = HfApi(token=token)
    try:
        all_files = api.list_repo_files(repo_id="roneneldan/TinyStories", repo_type="dataset")
    except Exception as e:
        print(f"    [Warning] Could not list repo files ({e}); using standard TinyStories paths.")
        all_files = [
            "data/validation-00000-of-00001-869c898b519ad725.parquet",
            "data/train-00000-of-00004-2d5a5a9018c64264.parquet",
        ]

    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    val_remote = [f for f in parquet_files if "validation" in f or "val" in f]
    train_remote = sorted([f for f in parquet_files if "train" in f])

    # 1 train shard has ~500M tokens; calculate how many shards needed
    shards_needed = max(1, math.ceil(needed_train_tokens / 500_000_000))
    train_remote = train_remote[:shards_needed]

    val_local = []
    for f in val_remote:
        print(f"    Downloading {f}...")
        p = download_hf_file("roneneldan/TinyStories", f, dest_dir, token=token)
        val_local.append(p)

    train_local = []
    for f in train_remote:
        print(f"    Downloading {f}...")
        p = download_hf_file("roneneldan/TinyStories", f, dest_dir, token=token)
        train_local.append(p)

    return val_local, train_local


def download_fineweb_edu(
    raw_dir: Path,
    token: Optional[str] = None,
    needed_tokens: int = 250_000_000,
) -> List[Path]:
    """Downloads only the required parquet shards from FineWeb-Edu (sample-10BT)."""
    dest_dir = raw_dir / "fineweb"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1 shard of sample-10BT contains ~100M tokens (~280 MB)
    shards_needed = max(1, math.ceil(needed_tokens / 100_000_000))

    existing = sorted(dest_dir.glob("**/*.parquet"))
    if len(existing) >= shards_needed:
        print(f"  * [FineWeb-Edu] Using local cache ({len(existing)} shards available, {shards_needed} needed)")
        return existing[:shards_needed]

    print(f"  * [FineWeb-Edu] Downloading {shards_needed} shard(s) (~{shards_needed * 280} MB) from HuggingFaceFW/fineweb-edu...")
    api = HfApi(token=token)
    try:
        all_files = api.list_repo_files(repo_id="HuggingFaceFW/fineweb-edu", repo_type="dataset")
        candidates = sorted([
            f for f in all_files
            if ("10BT" in f or "sample" in f) and f.endswith(".parquet")
        ])
        if not candidates:
            candidates = sorted([f for f in all_files if f.endswith(".parquet")])
    except Exception as e:
        print(f"    [Warning] Could not list repo files ({e}); using standard path pattern.")
        candidates = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(shards_needed)]

    selected = candidates[:shards_needed]
    local_files = []
    for f in selected:
        print(f"    Downloading {f}...")
        p = download_hf_file("HuggingFaceFW/fineweb-edu", f, dest_dir, token=token)
        local_files.append(p)

    return local_files


def download_simple_wikipedia(
    raw_dir: Path,
    token: Optional[str] = None,
) -> Any:
    """Downloads Simple English Wikipedia dataset locally via Hugging Face datasets."""
    dest_dir = raw_dir / "wiki"
    dest_dir.mkdir(parents=True, exist_ok=True)
    print("  * [Wikipedia] Checking/Downloading Simple English Wikipedia (~130 MB)...")
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.simple",
        split="train",
        token=token,
        cache_dir=str(dest_dir),
    )
    print(f"  * [Wikipedia] Loaded {len(ds):,} articles from local cache.")
    return ds


# ==============================================================================
# Local Document Streams
# ==============================================================================

def make_local_parquet_stream(
    files: List[Path],
    skip_docs: int = 0,
    min_len: int = 30,
) -> Iterator[str]:
    """Streams text documents from locally downloaded parquet files with batch chunking."""
    if not files:
        return

    skipped = 0
    while True:
        docs_yielded = 0
        for fpath in files:
            pf = pq.ParquetFile(str(fpath))
            for batch in pf.iter_batches(batch_size=2048, columns=["text"]):
                texts = batch.column("text").to_pylist()
                for text in texts:
                    if skipped < skip_docs:
                        skipped += 1
                        continue
                    if not text:
                        continue
                    t = text.strip()
                    if len(t) >= min_len:
                        docs_yielded += 1
                        yield t
        if docs_yielded == 0:
            break


def make_local_wikipedia_stream(
    dataset: Any,
    seed: int = 42,
) -> Iterator[str]:
    """Streams Simple English Wikipedia articles, cycling across epochs if needed."""
    epoch = 0
    while True:
        shuffled = dataset.shuffle(seed=seed + epoch)
        for row in shuffled:
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            if not text or len(text) < 40:
                continue
            doc = f"{title}\n\n{text}" if title else text
            yield doc
        epoch += 1


# ==============================================================================
# Balanced Token Stream Interleaver
# ==============================================================================

class BalancedStreamInterleaver:
    """Interleaves multiple document streams according to specified token ratios."""

    def __init__(
        self,
        streams: Dict[str, Iterator[str]],
        targets: Dict[str, int],
        tokenizer: tiktoken.Encoding,
    ):
        self.streams = streams
        self.targets = targets
        self.enc = tokenizer
        self.eot = tokenizer.eot_token

        # Only activate sources with positive target tokens
        self.tokens_collected = {k: 0 for k, v in targets.items() if v > 0}
        self.docs_collected = {k: 0 for k, v in targets.items() if v > 0}
        self.active_sources = set(self.tokens_collected.keys())

    def has_pending(self) -> bool:
        return any(
            self.tokens_collected[k] < self.targets[k]
            for k in self.active_sources
        )

    def next_document_tokens(self) -> Optional[Tuple[str, List[int]]]:
        """Selects the most under-represented dataset and returns its tokenized document."""
        eligible = [
            k for k in self.active_sources
            if self.tokens_collected[k] < self.targets[k]
        ]
        if not eligible:
            return None

        # Choose the source with the lowest completion fraction
        source = min(
            eligible,
            key=lambda k: self.tokens_collected[k] / max(1, self.targets[k]),
        )

        try:
            text = next(self.streams[source])
        except StopIteration:
            self.active_sources.remove(source)
            return self.next_document_tokens()

        # Tokenize with gpt2 tokenizer; append <|endoftext|> delimiter
        doc_tokens = self.enc.encode_ordinary(text)
        doc_tokens.append(self.eot)

        # Truncate if this document overshoots the source target
        remaining = self.targets[source] - self.tokens_collected[source]
        if len(doc_tokens) > remaining:
            doc_tokens = doc_tokens[:remaining]

        self.tokens_collected[source] += len(doc_tokens)
        self.docs_collected[source] += 1
        return source, doc_tokens


# ==============================================================================
# Corpus Preparation Function
# ==============================================================================

def prepare_split(
    split_name: str,
    output_dir: Path,
    target_tokens: int,
    ratios: Dict[str, float],
    streams: Dict[str, Iterator[str]],
    tokenizer: tiktoken.Encoding,
    chunk_flush_tokens: int = 1_000_000,
    shard_size_tokens: int = 0,
) -> Dict[str, Any]:
    """Generates packed uint16 binary files for a given split (train or val)."""
    targets = {
        name: int(round(target_tokens * ratio))
        for name, ratio in ratios.items()
    }
    # Adjust rounding differences to match exact total
    diff = target_tokens - sum(targets.values())
    if diff != 0:
        for k in ["fineweb", "wiki", "tinystories"]:
            if targets.get(k, 0) > 0:
                targets[k] += diff
                break

    print(f"\n--- Preparing '{split_name}' Split ({target_tokens:,} tokens) ---")
    for name, tgt in targets.items():
        pct = (ratios[name] * 100) if name in ratios else 0.0
        print(f"  * {name:12s}: {tgt:,} tokens ({pct:.1f}%)")

    # Filter streams to only those with target > 0
    active_streams = {k: v for k, v in streams.items() if targets.get(k, 0) > 0}
    interleaver = BalancedStreamInterleaver(active_streams, targets, tokenizer)

    use_shards = shard_size_tokens > 0 and shard_size_tokens < target_tokens
    shard_idx = 0
    current_shard_written = 0

    def get_shard_path(idx: int) -> Path:
        if use_shards:
            return output_dir / f"{split_name}_{idx:03d}.bin"
        return output_dir / f"{split_name}.bin"

    current_path = get_shard_path(shard_idx)
    if current_path.exists():
        current_path.unlink()

    token_buffer: List[int] = []
    total_written = 0

    pbar = tqdm(total=target_tokens, desc=f"Packing {split_name}", unit="tok")

    while interleaver.has_pending():
        res = interleaver.next_document_tokens()
        if res is None:
            break
        _, tokens = res
        token_buffer.extend(tokens)

        # Flush buffer in chunks to keep memory usage under 50 MB
        while len(token_buffer) >= chunk_flush_tokens:
            chunk = token_buffer[:chunk_flush_tokens]
            token_buffer = token_buffer[chunk_flush_tokens:]

            if use_shards and (current_shard_written + len(chunk) > shard_size_tokens):
                take = shard_size_tokens - current_shard_written
                overflow = chunk[take:]
                chunk = chunk[:take]

                arr = np.array(chunk, dtype=np.uint16)
                with open(current_path, "ab") as f:
                    f.write(arr.tobytes())
                total_written += len(chunk)
                pbar.update(len(chunk))

                shard_idx += 1
                current_path = get_shard_path(shard_idx)
                if current_path.exists():
                    current_path.unlink()
                current_shard_written = 0

                chunk = overflow

            arr = np.array(chunk, dtype=np.uint16)
            with open(current_path, "ab") as f:
                f.write(arr.tobytes())

            current_shard_written += len(chunk)
            total_written += len(chunk)
            pbar.update(len(chunk))

    # Flush remainder
    if token_buffer:
        arr = np.array(token_buffer, dtype=np.uint16)
        with open(current_path, "ab") as f:
            f.write(arr.tobytes())
        total_written += len(token_buffer)
        pbar.update(len(token_buffer))

    pbar.close()

    return {
        "total_tokens": total_written,
        "tokens_by_source": interleaver.tokens_collected,
        "docs_by_source": interleaver.docs_collected,
        "shards_count": shard_idx + 1 if use_shards else 1,
    }


# ==============================================================================
# Main CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare balanced pretraining corpus for JAX GPT-2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/data",
        help="Directory to save binary .bin files and metadata",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=None,
        help="Directory to cache raw downloaded parquet files (defaults to <output_dir>/raw)",
    )
    parser.add_argument(
        "--total_tokens",
        type=int,
        default=500_000_000,
        help="Total pretraining corpus tokens (e.g. 500M for 50M-70M GPT-2)",
    )
    parser.add_argument(
        "--val_tokens",
        type=int,
        default=5_000_000,
        help="Validation set tokens (e.g. 5M tokens = 1 percent)",
    )
    parser.add_argument(
        "--fineweb_ratio",
        type=float,
        default=0.50,
        help="Ratio of FineWeb-Edu tokens (default: 0.50)",
    )
    parser.add_argument(
        "--wiki_ratio",
        type=float,
        default=0.30,
        help="Ratio of Simple Wikipedia tokens (default: 0.30)",
    )
    parser.add_argument(
        "--tinystories_ratio",
        type=float,
        default=0.20,
        help="Ratio of TinyStories tokens (default: 0.20)",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=0,
        help="Tokens per shard file (0 = single train.bin and val.bin, e.g. 50_000_000 for 50M shards)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling and sampling",
    )
    parser.add_argument(
        "--chunk_flush_tokens",
        type=int,
        default=1_000_000,
        help="Buffer size in tokens before flushing to disk",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token (or read from HF_TOKEN / HUGGING_FACE_HUB_TOKEN env var)",
    )

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.fineweb_ratio + args.wiki_ratio + args.tinystories_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(f"Ratios must sum to 1.0, got: {total_ratio:.4f}")

    ratios = {
        "fineweb": args.fineweb_ratio,
        "wiki": args.wiki_ratio,
        "tinystories": args.tinystories_ratio,
    }

    hf_token = resolve_hf_token(args.hf_token)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(args.raw_dir) if args.raw_dir else (out_dir / "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("JAX GPT-2 Pretraining Dataset Preparation (Download-First Mode)")
    print(f"  Target Corpus:     {args.total_tokens:,} train tokens")
    print(f"  Validation Target: {args.val_tokens:,} val tokens")
    print(f"  Output Directory:  {out_dir.resolve()}")
    print(f"  Raw Cache Dir:     {raw_dir.resolve()}")
    if hf_token:
        print(f"  HF Authentication: Enabled (token: {hf_token[:4]}...{hf_token[-4:]})")
    else:
        print("  HF Authentication: None (set HF_TOKEN env var or --hf_token for faster downloads)")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # 1. Download Phase
    # --------------------------------------------------------------------------
    print("\n[Phase 1/3] Downloading/Verifying Datasets Locally...")
    dl_start = time.time()

    tinystories_val_files: List[Path] = []
    tinystories_train_files: List[Path] = []
    if ratios["tinystories"] > 0:
        needed_stories = int((args.total_tokens + args.val_tokens) * ratios["tinystories"])
        tinystories_val_files, tinystories_train_files = download_tinystories(
            raw_dir=raw_dir,
            token=hf_token,
            needed_train_tokens=needed_stories,
        )

    fineweb_files: List[Path] = []
    if ratios["fineweb"] > 0:
        needed_fineweb = int((args.total_tokens + args.val_tokens) * ratios["fineweb"])
        fineweb_files = download_fineweb_edu(
            raw_dir=raw_dir,
            token=hf_token,
            needed_tokens=needed_fineweb,
        )

    wiki_dataset: Any = None
    if ratios["wiki"] > 0:
        wiki_dataset = download_simple_wikipedia(
            raw_dir=raw_dir,
            token=hf_token,
        )

    dl_time = time.time() - dl_start
    print(f"[Phase 1/3 Complete] All data verified locally in {dl_time:.1f}s.")

    # --------------------------------------------------------------------------
    # 2. Tokenizer Initialization
    # --------------------------------------------------------------------------
    enc = tiktoken.get_encoding("gpt2")
    assert enc.n_vocab == 50257, f"Expected 50257 vocab, got {enc.n_vocab}"
    assert enc.eot_token == 50256, f"Expected 50256 EOT, got {enc.eot_token}"

    pack_start = time.time()

    # --------------------------------------------------------------------------
    # 3. Validation Split Packing
    # --------------------------------------------------------------------------
    val_streams: Dict[str, Iterator[str]] = {}
    if ratios["fineweb"] > 0 and fineweb_files:
        val_streams["fineweb"] = make_local_parquet_stream(
            fineweb_files, skip_docs=50_000, min_len=30
        )
    if ratios["wiki"] > 0 and wiki_dataset is not None:
        val_streams["wiki"] = make_local_wikipedia_stream(
            wiki_dataset, seed=args.seed + 999
        )
    if ratios["tinystories"] > 0 and tinystories_val_files:
        val_streams["tinystories"] = make_local_parquet_stream(
            tinystories_val_files, skip_docs=0, min_len=20
        )

    val_stats = prepare_split(
        split_name="val",
        output_dir=out_dir,
        target_tokens=args.val_tokens,
        ratios=ratios,
        streams=val_streams,
        tokenizer=enc,
        chunk_flush_tokens=min(args.chunk_flush_tokens, max(100_000, args.val_tokens // 2)),
        shard_size_tokens=args.shard_size,
    )

    # --------------------------------------------------------------------------
    # 4. Training Split Packing
    # --------------------------------------------------------------------------
    train_streams: Dict[str, Iterator[str]] = {}
    if ratios["fineweb"] > 0 and fineweb_files:
        train_streams["fineweb"] = make_local_parquet_stream(
            fineweb_files, skip_docs=0, min_len=30
        )
    if ratios["wiki"] > 0 and wiki_dataset is not None:
        train_streams["wiki"] = make_local_wikipedia_stream(
            wiki_dataset, seed=args.seed
        )
    if ratios["tinystories"] > 0 and tinystories_train_files:
        train_streams["tinystories"] = make_local_parquet_stream(
            tinystories_train_files, skip_docs=0, min_len=20
        )

    train_stats = prepare_split(
        split_name="train",
        output_dir=out_dir,
        target_tokens=args.total_tokens,
        ratios=ratios,
        streams=train_streams,
        tokenizer=enc,
        chunk_flush_tokens=args.chunk_flush_tokens,
        shard_size_tokens=args.shard_size,
    )

    pack_time = time.time() - pack_start
    total_time = dl_time + pack_time

    # --------------------------------------------------------------------------
    # 5. Metadata Writing
    # --------------------------------------------------------------------------
    meta = {
        "dataset_name": "jaxgpt_500m_mixture",
        "description": "Pretraining corpus for 50M-70M JAX GPT-2 (local download mode)",
        "tokenizer": "gpt2",
        "vocab_size": enc.n_vocab,
        "eot_token": enc.eot_token,
        "dtype": "uint16",
        "bytes_per_token": 2,
        "ratios_target": ratios,
        "train": train_stats,
        "val": val_stats,
        "download_seconds": round(dl_time, 2),
        "packing_seconds": round(pack_time, 2),
        "total_seconds": round(total_time, 2),
    }

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print("Dataset Preparation Completed Successfully!")
    print(f"  Train Tokens:     {train_stats['total_tokens']:,}")
    print(f"  Val Tokens:       {val_stats['total_tokens']:,}")
    print(f"  Download Time:    {dl_time:.1f}s")
    print(f"  Packing Time:     {pack_time:.1f}s")
    print(f"  Total Time:       {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  Metadata:         {meta_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
