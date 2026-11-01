"""Utilities package for JAX GPT-2: optimizers, learning rate schedulers, and checkpoints."""

from utils.checkpoint import (
    CheckpointManager,
    load_checkpoint,
    load_model_weights,
    save_checkpoint,
    save_model_weights,
)
from utils.lr_scheduler import (
    ScheduleFn,
    constant_schedule_with_warmup,
    cosine_schedule_with_warmup,
    create_lr_scheduler,
    linear_schedule_with_warmup,
    polynomial_schedule_with_warmup,
)
from utils.optimizer import (
    AdamW,
    AdamWState,
    adamw_step,
    clip_by_global_norm,
    get_default_decay_mask,
    init_adamw,
)

__all__ = [
    # Optimizer
    "AdamW",
    "AdamWState",
    "init_adamw",
    "adamw_step",
    "clip_by_global_norm",
    "get_default_decay_mask",
    # Schedulers
    "ScheduleFn",
    "cosine_schedule_with_warmup",
    "linear_schedule_with_warmup",
    "constant_schedule_with_warmup",
    "polynomial_schedule_with_warmup",
    "create_lr_scheduler",
    # Checkpointing
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "save_model_weights",
    "load_model_weights",
]
