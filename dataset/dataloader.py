"""High-performance, zero-copy memory-mapped DataLoader for JAX GPT-2."""

import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import jax
import numpy as np


class DataLoader:
    """Memory-mapped DataLoader for training autoregressive language models in JAX.

    Features:
      - Memory-maps binary token files (np.uint16) with zero RAM overhead.
      - Supports single .bin files, shard lists, or data directories.
      - Supports in-memory NumPy arrays.
      - Produces next-token prediction pairs:
          inputs:  tokens[:, :-1]  (shape: (B, T), dtype: int32)
          targets: tokens[:, 1:]   (shape: (B, T), dtype: int32)
      - shuffle=True: Uniform random window sampling for pretraining.
      - shuffle=False: Deterministic sequential evaluation for validation/perplexity.
      - return_jax=True: transfers batches directly to JAX device arrays.
      - Checkpoint support via state_dict() and load_state_dict().
    """

    def __init__(
        self,
        source: Union[str, Path, List[Union[str, Path]], np.ndarray],
        batch_size: int,
        seq_len: int,
        split: Optional[str] = None,
        shuffle: bool = True,
        seed: int = 42,
        return_jax: bool = False,
        dtype: np.dtype = np.uint16,
        drop_last: bool = True,
    ):
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.split = split
        self.shuffle = shuffle
        self.return_jax = return_jax
        self.dtype = np.dtype(dtype)
        self.drop_last = drop_last
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.shards: List[np.memmap] = []
        self.shard_lengths: List[int] = []
        self.cumulative_lengths: List[int] = []
        self.total_tokens: int = 0

        self._init_source(source)

        # Sequential iteration tracking
        self.current_idx = 0
        self.epoch = 0

    def _init_source(self, source: Any) -> None:
        """Resolves file paths, directories, or NumPy arrays into memory-mapped shards."""
        # 1. Directory or glob string
        if isinstance(source, (str, Path)):
            p = Path(source)
            if p.is_dir():
                pattern = f"*{self.split}*.bin" if self.split else "*.bin"
                files = sorted(p.glob(pattern))
                if not files:
                    files = sorted(p.glob("*.bin"))
                if not files:
                    raise FileNotFoundError(f"No binary token files found in directory: {p}")
                self._load_file_shards(files)
            elif "*" in str(source):
                files = sorted(glob.glob(str(source)))
                if not files:
                    raise FileNotFoundError(f"No files matched glob pattern: {source}")
                self._load_file_shards([Path(f) for f in files])
            elif p.is_file():
                self._load_file_shards([p])
            else:
                raise FileNotFoundError(f"Source file or directory does not exist: {source}")

        # 2. List of paths
        elif isinstance(source, (list, tuple)) and all(isinstance(x, (str, Path)) for x in source):
            paths = [Path(x) for x in source]
            for p in paths:
                if not p.is_file():
                    raise FileNotFoundError(f"Shard file not found: {p}")
            self._load_file_shards(paths)

        # 3. In-memory NumPy array
        elif isinstance(source, np.ndarray):
            if source.ndim != 1:
                source = source.ravel()
            self.total_tokens = len(source)
            self.shards = [source]
            self.shard_lengths = [self.total_tokens]
            self.cumulative_lengths = [self.total_tokens]

        else:
            raise TypeError(
                f"Unsupported source type: {type(source)}. "
                "Expected file path, list of paths, or numpy array."
            )

        min_needed = self.seq_len + 1
        if self.total_tokens < min_needed:
            raise ValueError(
                f"Dataset has {self.total_tokens:,} tokens, but requires at least "
                f"seq_len + 1 = {min_needed:,} tokens."
            )

    def _load_file_shards(self, file_paths: List[Path]) -> None:
        """Memory-maps multiple binary shards."""
        self.shards = []
        self.shard_lengths = []
        cum = 0

        for fp in file_paths:
            file_bytes = os.path.getsize(fp)
            itemsize = np.dtype(self.dtype).itemsize
            num_tokens = file_bytes // itemsize
            if num_tokens <= 0:
                continue

            mmap = np.memmap(fp, dtype=self.dtype, mode="r", shape=(num_tokens,))
            self.shards.append(mmap)
            self.shard_lengths.append(num_tokens)
            cum += num_tokens
            self.cumulative_lengths.append(cum)

        self.total_tokens = cum
        if not self.shards:
            raise ValueError(f"All shard files in {file_paths} were empty.")

    def __len__(self) -> int:
        """Total number of batches in a single pass."""
        tokens_per_batch = self.batch_size * self.seq_len
        return self.total_tokens // tokens_per_batch

    @property
    def num_tokens(self) -> int:
        """Total number of tokens in the dataset."""
        return self.total_tokens

    def _sample_slice(self, start_idx: int, length: int) -> np.ndarray:
        """Extracts a slice across one or multiple memory-mapped shards."""
        if len(self.shards) == 1:
            return self.shards[0][start_idx : start_idx + length]

        # Multi-shard case: find which shard start_idx belongs to
        shard_idx = int(np.searchsorted(self.cumulative_lengths, start_idx, side="right"))
        prev_cum = self.cumulative_lengths[shard_idx - 1] if shard_idx > 0 else 0
        local_start = start_idx - prev_cum
        shard_len = self.shard_lengths[shard_idx]

        if local_start + length <= shard_len:
            return self.shards[shard_idx][local_start : local_start + length]

        # Slice spans across a shard boundary: assemble from consecutive shards
        collected = []
        needed = length
        curr_shard = shard_idx
        curr_offset = local_start

        while needed > 0 and curr_shard < len(self.shards):
            available = self.shard_lengths[curr_shard] - curr_offset
            take = min(needed, available)
            collected.append(self.shards[curr_shard][curr_offset : curr_offset + take])
            needed -= take
            curr_shard += 1
            curr_offset = 0

        # If wrapped past end of last shard, wrap from the first shard
        if needed > 0:
            collected.append(self.shards[0][:needed])

        return np.concatenate(collected)

    def _next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fetches a batch from memory-mapped shards."""
        window_size = self.seq_len + 1
        max_start = self.total_tokens - window_size

        if self.shuffle:
            # Training: uniform random window positions across the full corpus
            starts = self.rng.integers(0, max_start + 1, size=self.batch_size)
            batch = np.stack(
                [self._sample_slice(idx, window_size) for idx in starts],
                axis=0,
            ).astype(np.int32)
        else:
            # Evaluation / Validation: deterministic sequential non-overlapping strides
            stride = self.seq_len
            last_start = self.current_idx + (self.batch_size - 1) * stride

            if last_start > max_start:
                if self.drop_last:
                    self.current_idx = 0
                    self.epoch += 1
                    raise StopIteration("End of dataset reached.")
                else:
                    self.current_idx = 0
                    self.epoch += 1

            starts = [self.current_idx + i * stride for i in range(self.batch_size)]
            self.current_idx += self.batch_size * stride

            batch = np.stack(
                [self._sample_slice(idx, window_size) for idx in starts],
                axis=0,
            ).astype(np.int32)

        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        return inputs, targets

    def next_batch(self) -> Tuple[Any, Any]:
        """Returns the next (inputs, targets) batch tuple of shape (B, T)."""
        inputs, targets = self._next_batch()

        if self.return_jax:
            return jax.device_put(inputs), jax.device_put(targets)
        return inputs, targets

    def __iter__(self) -> "DataLoader":
        if not self.shuffle:
            self.current_idx = 0
        return self

    def __next__(self) -> Tuple[Any, Any]:
        return self.next_batch()

    def reset(self) -> None:
        """Resets the sequential cursor and re-seeds RNG."""
        self.current_idx = 0
        self.epoch = 0
        self.rng = np.random.default_rng(self.seed)

    def state_dict(self) -> Dict[str, Any]:
        """Returns serializable state for exact training checkpoint resumption."""
        return {
            "current_idx": self.current_idx,
            "epoch": self.epoch,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restores DataLoader state from checkpoint."""
        self.current_idx = state.get("current_idx", 0)
        self.epoch = state.get("epoch", 0)
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]
