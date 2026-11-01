"""Pure JAX AdamW Optimizer with Decoupled Weight Decay and Gradient Clipping.

Implements the AdamW algorithm (Loshchilov & Hutter, 2019) natively in JAX:
- Decoupled weight decay: applied directly to parameters, avoiding gradient distortion.
- Weight decay filtering: automatically excludes 1D parameters (biases, LayerNorm scale/bias).
- Global gradient norm clipping: prevents catastrophic gradient explosions during LLM training.
- Fully compatible with `jax.jit`, JAX PyTrees, and variable learning rate schedulers.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Union
import jax
import jax.numpy as jnp
from jax import Array

from utils.lr_scheduler import ScheduleFn


class AdamWState(NamedTuple):
    """PyTree container for AdamW optimizer state.

    NamedTuple is natively registered as a JAX PyTree, enabling zero-boilerplate
    usage across `jax.jit` boundaries.
    """
    step: Array     # Scalar int32 representing current optimization step
    m: Any          # PyTree of first moment vectors (mean of gradients)
    v: Any          # PyTree of second moment vectors (uncentered variance of gradients)


# ---------------------------------------------------------------------------
# Gradient Clipping
# ---------------------------------------------------------------------------

def clip_by_global_norm(grads: Any, max_norm: float) -> Tuple[Any, Array]:
    """Clips the global L2 norm of gradients PyTree.

    Global norm is computed as:
        global_norm = sqrt(sum(||g||_2^2 for g in tree_leaves(grads)))

    If global_norm > max_norm:
        grads = grads * (max_norm / (global_norm + 1e-6))

    Args:
        grads: PyTree of gradient arrays.
        max_norm: Maximum allowed global L2 norm.

    Returns:
        Tuple of (clipped_grads, global_norm).
    """
    leaves = jax.tree_util.tree_leaves(grads)
    # Sum squared L2 norms over all leaves
    total_sq_norm = sum(jnp.sum(jnp.square(g)) for g in leaves)
    global_norm = jnp.sqrt(total_sq_norm)

    # Scaling factor
    scale = jnp.minimum(1.0, max_norm / (global_norm + 1e-6))
    clipped_grads = jax.tree_util.tree_map(lambda g: g * scale, grads)

    return clipped_grads, global_norm


# ---------------------------------------------------------------------------
# Default Weight Decay Filter
# ---------------------------------------------------------------------------

def get_default_decay_mask(params: Any) -> Any:
    """Generates a PyTree boolean mask for selective weight decay in Transformers.

    Weight decay rules for GPT-2 / LLMs:
      - 2D+ tensors (weight matrices, embedding tables): Decayed (True).
      - 1D tensors (biases, LayerNorm scale and bias vectors): Not decayed (False).

    Args:
        params: PyTree of model parameters.

    Returns:
        PyTree of booleans matching params structure.
    """
    return jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)


# ---------------------------------------------------------------------------
# Pure Functional AdamW Primitives
# ---------------------------------------------------------------------------

def init_adamw(params: Any) -> AdamWState:
    """Initializes AdamW state with zeros for 1st and 2nd moments.

    Args:
        params: PyTree of model parameters.

    Returns:
        AdamWState with step=0 and zero-initialized moment trees.
    """
    step = jnp.array(0, dtype=jnp.int32)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamWState(step=step, m=m, v=v)


def adamw_step(
    params: Any,
    grads: Any,
    state: AdamWState,
    learning_rate: Union[float, Array, ScheduleFn],
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    decay_mask: Optional[Any] = None,
    max_grad_norm: Optional[float] = 1.0,
) -> Tuple[Any, AdamWState, Dict[str, Array]]:
    """Performs a single AdamW optimization step.

    Formulation:
        1. step = state.step + 1
        2. g_clipped = clip(grads, max_grad_norm)
        3. m_t = b1 * m_{t-1} + (1 - b1) * g
        4. v_t = b2 * v_{t-1} + (1 - b2) * g^2
        5. m_hat = m_t / (1 - b1^step)
        6. v_hat = v_t / (1 - b2^step)
        7. p_new = p - lr * (m_hat / (sqrt(v_hat) + eps) + decay_term)

    Args:
        params: PyTree of model parameters.
        grads: PyTree of gradients with identical structure to params.
        state: Current AdamWState.
        learning_rate: Constant float/scalar or a callable schedule function `schedule(step) -> lr`.
        b1: First moment exponential decay factor (default: 0.9).
        b2: Second moment exponential decay factor (default: 0.999, or 0.95 for GPT-2).
        eps: Small constant for numerical stability (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 0.01).
        decay_mask: Optional boolean PyTree indicating which parameters receive weight decay.
            If None, automatically decays parameters with ndim >= 2 and spares 1D parameters.
        max_grad_norm: Maximum global norm for gradient clipping (default: 1.0).

    Returns:
        (new_params, new_state, metrics)
        where metrics contains scalar arrays: 'loss', 'grad_norm', 'lr', 'step'.
    """
    new_step = state.step + 1

    # 1. Resolve current learning rate
    if callable(learning_rate):
        lr = learning_rate(state.step)
    else:
        lr = jnp.asarray(learning_rate, dtype=jnp.float32)

    # 2. Global gradient norm clipping
    if max_grad_norm is not None and max_grad_norm > 0:
        clipped_grads, grad_norm = clip_by_global_norm(grads, float(max_grad_norm))
    else:
        clipped_grads = grads
        leaves = jax.tree_util.tree_leaves(grads)
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves))

    # 3. Default decay mask if not supplied
    if decay_mask is None:
        decay_mask = get_default_decay_mask(params)

    # 4. Bias correction factors
    step_f = jnp.asarray(new_step, dtype=jnp.float32)
    bias_correction1 = 1.0 - jnp.power(b1, step_f)
    bias_correction2 = 1.0 - jnp.power(b2, step_f)

    # 5. Moments update
    new_m = jax.tree_util.tree_map(
        lambda m_prev, g: b1 * m_prev + (1.0 - b1) * g,
        state.m,
        clipped_grads,
    )
    new_v = jax.tree_util.tree_map(
        lambda v_prev, g: b2 * v_prev + (1.0 - b2) * jnp.square(g),
        state.v,
        clipped_grads,
    )

    # 6. Parameter update with bias-correction and decoupled weight decay
    def update_param(p: Array, m_val: Array, v_val: Array, should_decay: bool) -> Array:
        m_hat = m_val / bias_correction1
        v_hat = v_val / bias_correction2
        adam_direction = m_hat / (jnp.sqrt(v_hat) + eps)
        decay_term = jnp.where(should_decay, weight_decay * p, 0.0)
        return p - lr * (adam_direction + decay_term)

    new_params = jax.tree_util.tree_map(
        update_param,
        params,
        new_m,
        new_v,
        decay_mask,
    )

    new_state = AdamWState(step=new_step, m=new_m, v=new_v)

    metrics = {
        "grad_norm": grad_norm,
        "lr": lr,
        "step": new_step,
    }

    return new_params, new_state, metrics


# ---------------------------------------------------------------------------
# Object-Oriented AdamW Optimizer Wrapper
# ---------------------------------------------------------------------------

class AdamW:
    """Configurable AdamW Optimizer class for JAX training loops.

    Encapsulates hyperparameters and provides `.init(params)` and `.step(params, grads, state)`.
    Fully compatible with `jax.jit`.
    """

    def __init__(
        self,
        learning_rate: Union[float, ScheduleFn] = 1e-3,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_grad_norm: Optional[float] = 1.0,
        decay_mask: Optional[Any] = None,
    ):
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.decay_mask = decay_mask

    def init(self, params: Any) -> AdamWState:
        """Initializes optimizer state."""
        return init_adamw(params)

    def step(
        self,
        params: Any,
        grads: Any,
        state: AdamWState,
    ) -> Tuple[Any, AdamWState, Dict[str, Array]]:
        """Applies a single gradient step and returns (new_params, new_state, metrics)."""
        return adamw_step(
            params=params,
            grads=grads,
            state=state,
            learning_rate=self.learning_rate,
            b1=self.b1,
            b2=self.b2,
            eps=self.eps,
            weight_decay=self.weight_decay,
            decay_mask=self.decay_mask,
            max_grad_norm=self.max_grad_norm,
        )

    def get_lr(self, step: Union[int, Array]) -> Array:
        """Returns the learning rate at a given step."""
        if callable(self.learning_rate):
            return self.learning_rate(step)
        return jnp.asarray(self.learning_rate, dtype=jnp.float32)
