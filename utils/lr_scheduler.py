"""Variable Learning Rate Schedulers for JAX Training.

All schedulers are implemented using pure JAX operations (jnp.where, jnp.clip, jnp.cos),
making them fully compatible with `jax.jit`, JAX tracers, and dynamic graph compilation.
"""

from __future__ import annotations

from typing import Callable, Union
import jax
import jax.numpy as jnp
from jax import Array


# Type alias for schedule functions: step -> learning_rate
ScheduleFn = Callable[[Union[int, Array]], Array]


def cosine_schedule_with_warmup(
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> ScheduleFn:
    """Cosine learning rate schedule with linear warmup.

    Schedule phases:
      1. Warmup (step < warmup_steps):
         Linearly ramps learning rate from `min_lr` to `peak_lr`.
      2. Cosine Decay (warmup_steps <= step <= total_steps):
         Decays learning rate following half-period cosine curve from `peak_lr` down to `min_lr`.
      3. Plateau (step > total_steps):
         Keeps learning rate at `min_lr`.

    Args:
        peak_lr: Maximum learning rate attained at the end of the warmup phase.
        warmup_steps: Number of steps for linear warmup (>= 0).
        total_steps: Total training steps at which learning rate reaches min_lr.
        min_lr: Final minimum learning rate. Defaults to 0.0 (or e.g. 10% of peak_lr).

    Returns:
        A JIT-compatible callable `schedule(step) -> lr`.
    """
    warmup_f = float(max(1, warmup_steps))
    total_f = float(max(warmup_steps + 1, total_steps))
    decay_steps_f = total_f - warmup_f

    def schedule(step: Union[int, Array]) -> Array:
        step_f = jnp.asarray(step, dtype=jnp.float32)

        # 1. Linear warmup: min_lr -> peak_lr
        warmup_ratio = jnp.clip(step_f / warmup_f, 0.0, 1.0)
        warmup_lr = min_lr + (peak_lr - min_lr) * warmup_ratio

        # 2. Cosine decay: peak_lr -> min_lr
        decay_ratio = jnp.clip((step_f - warmup_f) / decay_steps_f, 0.0, 1.0)
        cosine_coeff = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio))
        cosine_lr = min_lr + (peak_lr - min_lr) * cosine_coeff

        # Select phase based on step
        lr = jnp.where(step_f < warmup_steps, warmup_lr, cosine_lr)
        lr = jnp.where(step_f >= total_steps, min_lr, lr)
        return lr

    return schedule


def linear_schedule_with_warmup(
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> ScheduleFn:
    """Linear learning rate schedule with linear warmup and linear decay.

    Schedule phases:
      1. Warmup (step < warmup_steps):
         Linearly ramps learning rate from `min_lr` to `peak_lr`.
      2. Linear Decay (warmup_steps <= step <= total_steps):
         Linearly decays from `peak_lr` down to `min_lr`.
      3. Plateau (step > total_steps):
         Keeps learning rate at `min_lr`.

    Args:
        peak_lr: Maximum learning rate.
        warmup_steps: Steps for linear warmup.
        total_steps: Total training steps.
        min_lr: Minimum learning rate at end of decay.

    Returns:
        A JIT-compatible callable `schedule(step) -> lr`.
    """
    warmup_f = float(max(1, warmup_steps))
    total_f = float(max(warmup_steps + 1, total_steps))
    decay_steps_f = total_f - warmup_f

    def schedule(step: Union[int, Array]) -> Array:
        step_f = jnp.asarray(step, dtype=jnp.float32)

        # Warmup
        warmup_ratio = jnp.clip(step_f / warmup_f, 0.0, 1.0)
        warmup_lr = min_lr + (peak_lr - min_lr) * warmup_ratio

        # Linear decay
        decay_ratio = jnp.clip((step_f - warmup_f) / decay_steps_f, 0.0, 1.0)
        decay_lr = peak_lr - (peak_lr - min_lr) * decay_ratio

        lr = jnp.where(step_f < warmup_steps, warmup_lr, decay_lr)
        lr = jnp.where(step_f >= total_steps, min_lr, lr)
        return lr

    return schedule


def constant_schedule_with_warmup(
    peak_lr: float,
    warmup_steps: int,
    min_lr: float = 0.0,
) -> ScheduleFn:
    """Constant learning rate schedule with linear warmup.

    Ramps up linearly to `peak_lr` and remains at `peak_lr` indefinitely.

    Args:
        peak_lr: Target constant learning rate.
        warmup_steps: Warmup step count.
        min_lr: Initial learning rate at step 0.

    Returns:
        A JIT-compatible callable `schedule(step) -> lr`.
    """
    warmup_f = float(max(1, warmup_steps))

    def schedule(step: Union[int, Array]) -> Array:
        step_f = jnp.asarray(step, dtype=jnp.float32)
        warmup_ratio = jnp.clip(step_f / warmup_f, 0.0, 1.0)
        warmup_lr = min_lr + (peak_lr - min_lr) * warmup_ratio
        return jnp.where(step_f < warmup_steps, warmup_lr, peak_lr)

    return schedule


def polynomial_schedule_with_warmup(
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    power: float = 1.0,
    min_lr: float = 0.0,
) -> ScheduleFn:
    """Polynomial decay learning rate schedule with linear warmup.

    Decays according to: `(1 - progress) ** power`.

    Args:
        peak_lr: Maximum learning rate.
        warmup_steps: Warmup step count.
        total_steps: Total steps.
        power: Exponent for decay polynomial (e.g. 2.0 = quadratic). Default 1.0.
        min_lr: Final minimum learning rate.

    Returns:
        A JIT-compatible callable `schedule(step) -> lr`.
    """
    warmup_f = float(max(1, warmup_steps))
    total_f = float(max(warmup_steps + 1, total_steps))
    decay_steps_f = total_f - warmup_f

    def schedule(step: Union[int, Array]) -> Array:
        step_f = jnp.asarray(step, dtype=jnp.float32)

        warmup_ratio = jnp.clip(step_f / warmup_f, 0.0, 1.0)
        warmup_lr = min_lr + (peak_lr - min_lr) * warmup_ratio

        decay_ratio = jnp.clip((step_f - warmup_f) / decay_steps_f, 0.0, 1.0)
        poly_lr = min_lr + (peak_lr - min_lr) * ((1.0 - decay_ratio) ** power)

        lr = jnp.where(step_f < warmup_steps, warmup_lr, poly_lr)
        lr = jnp.where(step_f >= total_steps, min_lr, lr)
        return lr

    return schedule


def create_lr_scheduler(
    schedule_type: str = "cosine",
    peak_lr: float = 6e-4,
    warmup_steps: int = 2000,
    total_steps: int = 100000,
    min_lr: float = 6e-5,
    power: float = 1.0,
) -> ScheduleFn:
    """Convenience factory function to instantiate a variable learning rate schedule.

    Supported types: 'cosine', 'linear', 'constant', 'polynomial'.

    Args:
        schedule_type: Name of schedule type.
        peak_lr: Maximum learning rate.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps for decay.
        min_lr: Minimum floor learning rate.
        power: Exponent for polynomial decay.

    Returns:
        JIT-compatible schedule function.
    """
    st = schedule_type.lower().strip()
    if st == "cosine":
        return cosine_schedule_with_warmup(
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
        )
    elif st == "linear":
        return linear_schedule_with_warmup(
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
        )
    elif st == "constant":
        return constant_schedule_with_warmup(
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
        )
    elif st in ("poly", "polynomial"):
        return polynomial_schedule_with_warmup(
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            power=power,
            min_lr=min_lr,
        )
    else:
        raise ValueError(
            f"Unknown schedule_type '{schedule_type}'. "
            "Supported types: 'cosine', 'linear', 'constant', 'polynomial'."
        )

