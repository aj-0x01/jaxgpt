"""Pure JAX Causal Multi-Head Self-Attention for GPT-2 architectures.

This module implements Causal Multi-Head Self-Attention (MHA) using strictly pure JAX
(jax, jax.numpy, jax.random, jax.nn) with zero external framework dependencies
(no Flax, Haiku, or Equinox).

Key Features:
- Fused QKV linear projection (c_attn) matching OpenAI GPT-2 architecture.
- Efficient head splitting, transposition, and batched matrix multiplications.
- Causal masking ensuring autoregressive attention (no future token leakage).
- Numerically stable scaled dot-product attention via jax.nn.softmax.
- Optional dropout during training; no-op during deterministic evaluation.
- Pure functional API (`init_causal_self_attention`, `causal_self_attention`)
  fully compatible with `jax.jit`, `jax.grad`, and `jax.vmap`.
- Lightweight object-oriented `CausalSelfAttention` wrapper class.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array


# Type alias for parameter trees
Params = Dict[str, Dict[str, Array]]


class AttentionConfig(NamedTuple):
    """Configuration for Causal Multi-Head Self-Attention."""
    d_model: int
    n_heads: int
    max_seq_len: int = 1024
    dropout_rate: float = 0.0
    bias: bool = True
    initializer_std: float = 0.02
    # Scaling factor for the residual output projection (e.g. 1 / sqrt(2 * n_layer))
    proj_scale: Optional[float] = None


def init_causal_self_attention(
    rng: Array,
    d_model: int,
    n_heads: int,
    max_seq_len: int = 1024,
    bias: bool = True,
    initializer_std: float = 0.02,
    n_layer: Optional[int] = None,
    proj_scale: Optional[float] = None,
    dtype: jnp.dtype = jnp.float32,
) -> Params:
    """Initializes parameters for Causal Multi-Head Self-Attention.

    Uses fused projection for Query, Key, and Value projections (c_attn) as in
    OpenAI GPT-2, followed by an output projection (c_proj).

    Weights are drawn from a normal distribution with standard deviation
    `initializer_std`. The output projection can optionally be scaled by
    `proj_scale` (typically 1 / sqrt(2 * n_layer) as in the GPT-2 paper).

    Args:
        rng: JAX PRNGKey for pseudo-random number generation.
        d_model: Total embedding / model dimension.
        n_heads: Number of attention heads (d_model must be divisible by n_heads).
        max_seq_len: Maximum sequence length (used for validation).
        bias: Whether to include bias terms in linear projections.
        initializer_std: Base standard deviation for weight initialization.
        n_layer: Number of layers in model (used for residual projection scaling 1 / sqrt(2 * n_layer)).
        proj_scale: Optional multiplier for the output projection weight standard deviation.
        dtype: JAX array data type (defaults to float32).

    Returns:
        A nested dictionary of JAX arrays (PyTree) representing layer parameters:
        {
            "c_attn": {"w": (d_model, 3 * d_model), "b": (3 * d_model,)},
            "c_proj": {"w": (d_model, d_model), "b": (d_model,)},
        }
    """
    if d_model % n_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

    rng_attn, rng_proj = jax.random.split(rng, 2)

    # Fused QKV projection weights: (d_model, 3 * d_model)
    w_attn = jax.random.normal(
        rng_attn, shape=(d_model, 3 * d_model), dtype=dtype
    ) * initializer_std

    # Output projection weights: (d_model, d_model)
    proj_std = initializer_std
    if proj_scale is not None:
        proj_std *= proj_scale
    elif n_layer is not None and n_layer > 0:
        proj_std /= (2.0 * float(n_layer)) ** 0.5

    w_proj = jax.random.normal(
        rng_proj, shape=(d_model, d_model), dtype=dtype
    ) * proj_std

    params: Params = {
        "c_attn": {"w": w_attn},
        "c_proj": {"w": w_proj},
    }

    if bias:
        params["c_attn"]["b"] = jnp.zeros((3 * d_model,), dtype=dtype)
        params["c_proj"]["b"] = jnp.zeros((d_model,), dtype=dtype)

    return params


def causal_self_attention(
    params: Params,
    x: Array,
    n_heads: int,
    mask: Optional[Array] = None,
    deterministic: bool = True,
    rng: Optional[Array] = None,
    dropout_rate: float = 0.0,
    return_attn_weights: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    """Pure functional forward pass for Causal Multi-Head Self-Attention.

    Args:
        params: Dictionary containing 'c_attn' and 'c_proj' weight/bias arrays.
        x: Input tensor of shape (batch_size, seq_len, d_model).
        n_heads: Number of attention heads.
        mask: Optional custom boolean or additive mask. If None, standard lower-triangular
              causal mask is generated dynamically based on seq_len.
        deterministic: If True, dropout is disabled (inference mode).
        rng: PRNGKey required if deterministic=False and dropout_rate > 0.
        dropout_rate: Probability of dropping attention weights.
        return_attn_weights: If True, returns tuple (output, attn_weights).

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model).
        If `return_attn_weights=True`, returns (output, attention_weights)
        where attention_weights has shape (batch_size, n_heads, seq_len, seq_len).
    """
    batch_size, seq_len, d_model = x.shape
    d_head = d_model // n_heads

    # 1. Fused QKV Projection: (B, T, d_model) @ (d_model, 3 * d_model) -> (B, T, 3 * d_model)
    w_attn = params["c_attn"].get("w", params["c_attn"].get("weight"))
    b_attn = params["c_attn"].get("b", params["c_attn"].get("bias"))
    qkv = jnp.matmul(x, w_attn)
    if b_attn is not None:
        qkv = qkv + b_attn

    # Split into Query, Key, Value tensors: each of shape (B, T, d_model)
    q, k, v = jnp.split(qkv, 3, axis=-1)

    # 2. Reshape and Transpose for Multi-Head Attention:
    # (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
    q = q.reshape(batch_size, seq_len, n_heads, d_head).swapaxes(1, 2)
    k = k.reshape(batch_size, seq_len, n_heads, d_head).swapaxes(1, 2)
    v = v.reshape(batch_size, seq_len, n_heads, d_head).swapaxes(1, 2)

    # 3. Scaled Dot-Product Attention:
    # Q @ K^T: (B, nh, T, dh) @ (B, nh, dh, T) -> (B, nh, T, T)
    scale = jnp.array(1.0 / (d_head ** 0.5), dtype=x.dtype)
    attn_scores = jnp.matmul(q, k.swapaxes(-1, -2)) * scale

    # 4. Causal Masking:
    # Ensure token at position i only attends to positions j <= i
    if mask is None:
        # Standard causal lower triangular mask (T, T)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        # Mask future tokens with large negative value (-1e9)
        attn_scores = jnp.where(causal_mask, attn_scores, -1e9)
    else:
        # If external mask is provided:
        # If boolean mask: True means keep, False means mask out
        if mask.dtype == jnp.bool_:
            attn_scores = jnp.where(mask, attn_scores, -1e9)
        else:
            # Additive mask (e.g. 0.0 for keep, -1e9 for mask)
            attn_scores = attn_scores + mask

    # 5. Numerically stable Softmax over key sequence dimension
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)

    # 6. Attention Dropout (if active)
    if (not deterministic) and (dropout_rate > 0.0):
        if rng is None:
            raise ValueError("rng must be provided when deterministic=False and dropout_rate > 0.0")
        keep_prob = 1.0 - dropout_rate
        dropout_mask = jax.random.bernoulli(rng, p=keep_prob, shape=attn_weights.shape)
        attn_weights = jnp.where(dropout_mask, attn_weights / keep_prob, 0.0)

    # 7. Context aggregation:
    # (B, nh, T, T) @ (B, nh, T, dh) -> (B, nh, T, dh)
    out = jnp.matmul(attn_weights, v)

    # 8. Recombine heads:
    # (B, nh, T, dh) -> (B, T, nh, dh) -> (B, T, d_model)
    out = out.swapaxes(1, 2).reshape(batch_size, seq_len, d_model)

    # 9. Output projection (c_proj)
    w_proj = params["c_proj"].get("w", params["c_proj"].get("weight"))
    b_proj = params["c_proj"].get("b", params["c_proj"].get("bias"))
    out = jnp.matmul(out, w_proj)
    if b_proj is not None:
        out = out + b_proj

    if return_attn_weights:
        return out, attn_weights
    return out


class CausalSelfAttention:
    """Object-oriented wrapper for Causal Multi-Head Self-Attention.

    Encapsulates hyperparameters while maintaining a pure functional paradigm
    where parameters are kept as explicit JAX PyTrees.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 1024,
        dropout_rate: float = 0.0,
        bias: bool = True,
        initializer_std: float = 0.02,
        proj_scale: Optional[float] = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.bias = bias
        self.initializer_std = initializer_std
        self.proj_scale = proj_scale
        self.dtype = dtype

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

    def init(self, rng: Array) -> Params:
        """Initialize parameters."""
        return init_causal_self_attention(
            rng=rng,
            d_model=self.d_model,
            n_heads=self.n_heads,
            max_seq_len=self.max_seq_len,
            bias=self.bias,
            initializer_std=self.initializer_std,
            proj_scale=self.proj_scale,
            dtype=self.dtype,
        )

    def __call__(
        self,
        params: Params,
        x: Array,
        mask: Optional[Array] = None,
        deterministic: bool = True,
        rng: Optional[Array] = None,
        return_attn_weights: bool = False,
    ) -> Union[Array, Tuple[Array, Array]]:
        """Run forward attention pass."""
        return causal_self_attention(
            params=params,
            x=x,
            n_heads=self.n_heads,
            mask=mask,
            deterministic=deterministic,
            rng=rng,
            dropout_rate=self.dropout_rate,
            return_attn_weights=return_attn_weights,
        )
