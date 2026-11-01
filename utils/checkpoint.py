"""Pure JAX Checkpoint Management for GPT-2 Models and Optimizers.

Provides robust, atomic serialization and deserialization for:
- Model parameters (PyTree of JAX arrays)
- Optimizer state (AdamWState: step, 1st & 2nd moments)
- Model architecture configuration (GPTConfig)
- DataLoader cursors (resuming exact token offsets)
- Training metrics and step metadata

Features:
- Atomic writes: writes to a temporary file and atomically replaces it to prevent corruption.
- CheckpointManager: automatic pruning (keeps last K checkpoints + best checkpoint by loss).
- Model-only export: lightweight saves for inference/eval without optimizer overhead.
- Human-readable JSON metadata alongside binary checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import glob
import json
import os
from pathlib import Path
import pickle
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

from dataset.dataloader import DataLoader
from gpt.gpt import GPT2, GPTConfig
from utils.optimizer import AdamWState


# ---------------------------------------------------------------------------
# Helper Functions: Host / Device Transfers
# ---------------------------------------------------------------------------

def _to_host(pytree: Any) -> Any:
    """Converts all JAX device arrays in a PyTree to CPU NumPy arrays."""
    return jax.device_get(pytree)


def _to_device(pytree: Any) -> Any:
    """Converts all NumPy arrays in a PyTree to JAX device arrays."""
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x) if hasattr(x, "__array__") else x,
        pytree,
    )


# ---------------------------------------------------------------------------
# Save & Load Primitives
# ---------------------------------------------------------------------------

def save_checkpoint(
    checkpoint_path: Union[str, Path],
    model_params: Mapping[str, Any],
    opt_state: Optional[AdamWState] = None,
    config: Optional[GPTConfig] = None,
    step: int = 0,
    epoch: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    dataloader: Optional[DataLoader] = None,
    save_meta_json: bool = True,
) -> Path:
    """Atomically saves a complete training checkpoint.

    Args:
        checkpoint_path: Target path for the checkpoint file (e.g. 'artifacts/ckpt_step_1000.ckpt').
        model_params: PyTree of model parameters.
        opt_state: AdamWState optimizer state (optional).
        config: GPTConfig model configuration (optional).
        step: Current training step.
        epoch: Current training epoch.
        metrics: Dictionary of current metrics (e.g. {'train_loss': 3.2, 'val_loss': 3.4, 'lr': 1e-4}).
        dataloader: Active DataLoader instance to save cursor / shard progress.
        save_meta_json: Whether to write an adjacent human-readable `.json` metadata file.

    Returns:
        Resolved Path to the saved checkpoint.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Prepare data dictionary
    data: Dict[str, Any] = {
        "step": int(step),
        "epoch": int(epoch),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_params": _to_host(model_params),
        "metrics": metrics or {},
    }

    if opt_state is not None:
        data["opt_state"] = {
            "step": int(opt_state.step),
            "m": _to_host(opt_state.m),
            "v": _to_host(opt_state.v),
        }

    if config is not None:
        data["config"] = asdict(config)

    if dataloader is not None:
        data["dataloader_state"] = dataloader.state_dict()

    # 2. Atomic write: write to temp file then atomic rename
    tmp_path = path.with_suffix(path.suffix + f".tmp_{os.getpid()}")
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # 3. Optional human-readable metadata JSON alongside checkpoint
    if save_meta_json:
        meta_dict = {
            "step": int(step),
            "epoch": int(epoch),
            "timestamp": data["timestamp"],
            "metrics": metrics or {},
            "checkpoint_file": path.name,
            "has_optimizer": opt_state is not None,
            "has_dataloader": dataloader is not None,
            "config": asdict(config) if config is not None else None,
        }
        meta_json_path = path.with_suffix(".json")
        with open(meta_json_path, "w") as f:
            json.dump(meta_dict, f, indent=2)

    return path.resolve()


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    restore_to_device: bool = True,
    dataloader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    """Loads a training checkpoint from disk.

    Args:
        checkpoint_path: Path to the .ckpt or .pkl file.
        restore_to_device: If True, transfers all arrays directly to JAX device arrays.
        dataloader: Optional DataLoader instance whose state will be restored if saved.

    Returns:
        Dictionary containing:
            - 'model_params': PyTree of model parameters (restored to JAX arrays if restore_to_device=True)
            - 'opt_state': AdamWState (if present in checkpoint)
            - 'config': GPTConfig (if present)
            - 'step': int
            - 'epoch': int
            - 'metrics': Dict[str, Any]
            - 'dataloader_state': Dict[str, Any] (if present)
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    result: Dict[str, Any] = {
        "step": data.get("step", 0),
        "epoch": data.get("epoch", 0),
        "metrics": data.get("metrics", {}),
        "timestamp": data.get("timestamp"),
    }

    # 1. Restore Model Parameters
    model_params = data["model_params"]
    if restore_to_device:
        model_params = _to_device(model_params)
    result["model_params"] = model_params

    # 2. Restore Optimizer State
    if "opt_state" in data:
        opt_dict = data["opt_state"]
        m = _to_device(opt_dict["m"]) if restore_to_device else opt_dict["m"]
        v = _to_device(opt_dict["v"]) if restore_to_device else opt_dict["v"]
        step_arr = jnp.array(opt_dict["step"], dtype=jnp.int32)
        result["opt_state"] = AdamWState(step=step_arr, m=m, v=v)
    else:
        result["opt_state"] = None

    # 3. Restore Config
    if "config" in data:
        result["config"] = GPTConfig(**data["config"])
    else:
        result["config"] = None

    # 4. Restore DataLoader state
    if "dataloader_state" in data:
        result["dataloader_state"] = data["dataloader_state"]
        if dataloader is not None:
            dataloader.load_state_dict(data["dataloader_state"])
    else:
        result["dataloader_state"] = None

    return result


# ---------------------------------------------------------------------------
# Lightweight Model-Only Export (for Inference / Deployment)
# ---------------------------------------------------------------------------

def save_model_weights(
    weights_path: Union[str, Path],
    model_params: Mapping[str, Any],
    config: GPTConfig,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Saves lightweight model parameters and config only (omits optimizer states).

    Useful for sharing weights, deploying models, or running inference without
    the 2-3x memory overhead of optimizer moment tensors.
    """
    return save_checkpoint(
        checkpoint_path=weights_path,
        model_params=model_params,
        opt_state=None,
        config=config,
        step=0,
        metrics=metrics,
        save_meta_json=True,
    )


def load_model_weights(
    weights_path: Union[str, Path],
    restore_to_device: bool = True,
) -> Tuple[Dict[str, Any], GPTConfig]:
    """Loads lightweight model parameters and GPTConfig.

    Returns:
        (model_params, config)
    """
    ckpt = load_checkpoint(weights_path, restore_to_device=restore_to_device)
    if ckpt["config"] is None:
        raise ValueError(f"No GPTConfig found in checkpoint: {weights_path}")
    return ckpt["model_params"], ckpt["config"]


# ---------------------------------------------------------------------------
# Checkpoint Manager (Pruning, Best Checkpoint Tracking, Auto-Resume)
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Manages saving, pruning, and resuming training checkpoints in a directory.

    Features:
    - Maintains the last `max_to_keep` step checkpoints (e.g. keeps 3 most recent).
    - Automatically tracks and preserves `best_model.ckpt` based on lowest validation loss.
    - Atomically writes checkpoints to prevent corruption.
    - Provides `latest_checkpoint()` and `best_checkpoint()` lookup.
    """

    def __init__(
        self,
        directory: Union[str, Path] = "artifacts/checkpoints",
        max_to_keep: int = 3,
        metric_name: str = "val_loss",
        mode: str = "min",
    ):
        """
        Args:
            directory: Directory to store checkpoint files.
            max_to_keep: Maximum number of recent step checkpoints to retain (older ones pruned).
            metric_name: Key in metrics dictionary to monitor for best checkpoint (e.g. 'val_loss').
            mode: 'min' if lower metric is better (e.g. loss), 'max' if higher is better (e.g. accuracy).
        """
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.metric_name = metric_name
        self.mode = mode.lower()
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got: {mode}")

        self.best_metric_value: Optional[float] = None
        self._history: List[Tuple[int, Path]] = []  # List of (step, path) sorted by step

        self._scan_existing_checkpoints()

    def _scan_existing_checkpoints(self) -> None:
        """Scans directory for existing step checkpoints and best model."""
        pattern = str(self.directory / "ckpt_step_*.ckpt")
        found = glob.glob(pattern)
        self._history = []

        for p_str in sorted(found):
            p = Path(p_str)
            try:
                # Extract step number from name: ckpt_step_0001000.ckpt
                parts = p.stem.split("_")
                step_idx = int(parts[-1])
                self._history.append((step_idx, p))
            except (ValueError, IndexError):
                continue

        self._history.sort(key=lambda x: x[0])

    def save(
        self,
        step: int,
        model_params: Mapping[str, Any],
        opt_state: Optional[AdamWState] = None,
        config: Optional[GPTConfig] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        dataloader: Optional[DataLoader] = None,
    ) -> Path:
        """Saves a checkpoint for the current step, updates best model, and prunes older ones.

        Args:
            step: Current training step integer.
            model_params: Model parameter PyTree.
            opt_state: Current optimizer state.
            config: Model GPTConfig.
            epoch: Current training epoch.
            metrics: Dictionary of metrics (e.g. {'train_loss': 3.1, 'val_loss': 3.3}).
            dataloader: DataLoader instance.

        Returns:
            Path to the saved step checkpoint.
        """
        metrics = metrics or {}

        # 1. Save standard step checkpoint
        ckpt_filename = f"ckpt_step_{step:08d}.ckpt"
        ckpt_path = self.directory / ckpt_filename
        saved_path = save_checkpoint(
            checkpoint_path=ckpt_path,
            model_params=model_params,
            opt_state=opt_state,
            config=config,
            step=step,
            epoch=epoch,
            metrics=metrics,
            dataloader=dataloader,
            save_meta_json=True,
        )

        self._history.append((step, saved_path))
        self._history.sort(key=lambda x: x[0])

        # 2. Check if this is the best checkpoint so far
        current_metric = metrics.get(self.metric_name)
        if current_metric is not None:
            current_val = float(current_metric)
            is_best = False
            if self.best_metric_value is None:
                is_best = True
            elif self.mode == "min" and current_val < self.best_metric_value:
                is_best = True
            elif self.mode == "max" and current_val > self.best_metric_value:
                is_best = True

            if is_best:
                self.best_metric_value = current_val
                best_path = self.directory / "best_model.ckpt"
                save_checkpoint(
                    checkpoint_path=best_path,
                    model_params=model_params,
                    opt_state=opt_state,
                    config=config,
                    step=step,
                    epoch=epoch,
                    metrics={**metrics, "is_best": True},
                    dataloader=dataloader,
                    save_meta_json=True,
                )

        # 3. Prune old checkpoints exceeding max_to_keep
        self._prune()

        return saved_path

    def _prune(self) -> None:
        """Removes oldest step checkpoints exceeding max_to_keep."""
        while len(self._history) > self.max_to_keep:
            oldest_step, oldest_path = self._history.pop(0)
            try:
                if oldest_path.exists():
                    oldest_path.unlink()
                json_path = oldest_path.with_suffix(".json")
                if json_path.exists():
                    json_path.unlink()
            except OSError:
                pass

    def latest_checkpoint(self) -> Optional[Path]:
        """Returns path to the most recent step checkpoint, or None."""
        if not self._history:
            self._scan_existing_checkpoints()
        if self._history:
            return self._history[-1][1]
        return None

    def best_checkpoint(self) -> Optional[Path]:
        """Returns path to best_model.ckpt if it exists, or None."""
        p = self.directory / "best_model.ckpt"
        return p if p.is_file() else None

    def restore_latest_or_init(
        self,
        default_config: GPTConfig,
        init_rng: Array,
        dataloader: Optional[DataLoader] = None,
    ) -> Tuple[Dict[str, Any], Optional[AdamWState], GPTConfig, int]:
        """Convenience method: resumes from latest checkpoint if available, or initializes anew.

        Returns:
            (model_params, opt_state, config, start_step)
        """
        latest = self.latest_checkpoint()
        if latest is not None:
            ckpt = load_checkpoint(latest, restore_to_device=True, dataloader=dataloader)
            return (
                ckpt["model_params"],
                ckpt["opt_state"],
                ckpt["config"] or default_config,
                ckpt["step"],
            )

        # Initialize fresh model
        from gpt.gpt import init_gpt2
        params = init_gpt2(init_rng, default_config)
        return params, None, default_config, 0
