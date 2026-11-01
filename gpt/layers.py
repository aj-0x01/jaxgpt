"""Pure JAX implementation of Layer Normalization and MLP/FFN for GPT-2 architectures.

This module implements:
1. Layer Normalization (LayerNorm):
   - Learnable scale (gamma/weight) initialized to 1.0.
   - Learnable bias (beta/bias) initialized to 0.0.
   - Epsilon eps=1e-5.
   - Mean and variance over the last feature dimension (-1).
   - Upcasts half-precision inputs to float32 during normalization for numerical stability.

2. MLP / Feed Forward Network (FFN):
   - In GPT-2, MLP and Feed Forward Network are the same concept:
     x -> c_fc (d_model -> 4*d_model) -> GELU -> c_proj (4*d_model -> d_model) -> out
   - Support for both exact GELU and the GPT-2 tanh-approximation.
   - Initializer std=0.02 with optional residual projection scaling 1 / sqrt(2 * n_layer).
   - Functional API and full alias equivalence: `feed_forward = mlp`.

Strictly pure JAX (jax, jax.numpy, jax.random, jax.nn) with zero external frameworks
(no Flax, Haiku, or Equinox). Fully compatible with jax.jit, jax.grad, and PyTrees.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, NamedTuple, Optional, Tuple, TypedDict, Union
import jax
import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# Typing & Config Definitions
# ---------------------------------------------------------------------------

class LayerNormParams(TypedDict, total=False):
    """Parameter dictionary structure for Layer Normalization."""
    scale: Array  # Learnable scale (gamma / weight) of shape (d_model,)
    bias: Array   # Learnable bias (beta) of shape (d_model,)


class LinearParams(TypedDict, total=False):
    """Parameter dictionary structure for a linear projection."""
    w: Array      # Weight matrix of shape (in_features, out_features)
    bias: Array   # Bias vector of shape (out_features,)
    b: Array      # Optional alias for bias


class MLPParams(TypedDict, total=False):
    """Parameter dictionary structure for GPT-2 MLP / Feed-Forward Network."""
    c_fc: LinearParams    # Input expansion: d_model -> 4*d_model
    c_proj: LinearParams  # Residual projection: 4*d_model -> d_model


# FFN aliases
FFNParams = MLPParams


class LayerNormConfig(NamedTuple):
    """Configuration for Layer Normalization."""
    d_model: int
    eps: float = 1e-5
    dtype: jnp.dtype = jnp.float32


class MLPConfig(NamedTuple):
    """Configuration for GPT-2 MLP / Feed-Forward Network."""
    d_model: int
    d_ff: Optional[int] = None
    bias: bool = True
    initializer_std: float = 0.02
    n_layer: Optional[int] = None
    proj_scale: Optional[float] = None
    approximate: bool = True
    dtype: jnp.dtype = jnp.float32


FeedForwardConfig = MLPConfig


# ---------------------------------------------------------------------------
# GELU Activation
# ---------------------------------------------------------------------------

def gelu(x: Array, approximate: bool = True) -> Array:
    """Gaussian Error Linear Unit (GELU) activation.

    Supports both the exact formulation and the tanh approximation used by GPT-2.

    Exact formula:
        GELU(x) = x * Phi(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

    GPT-2 Tanh approximation (Radford et al., 2019):
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

    Args:
        x: Input tensor of arbitrary shape.
        approximate: Whether to use the GPT-2 tanh approximation (True) or exact (False).
            Defaults to True for strict GPT-2 fidelity.

    Returns:
        Tensor of same shape and dtype as x.
    """
    if approximate:
        # sqrt(2 / pi) ≈ 0.7978845608028654
        sqrt_2_over_pi = 0.7978845608028654
        return 0.5 * x * (1.0 + jnp.tanh(sqrt_2_over_pi * (x + 0.044715 * jnp.power(x, 3))))
    return jax.nn.gelu(x, approximate=False)


# ---------------------------------------------------------------------------
# Layer Normalization
# ---------------------------------------------------------------------------

def init_layer_norm(
    d_model: int,
    eps: float = 1e-5,
    dtype: jnp.dtype = jnp.float32,
) -> LayerNormParams:
    """Initialize parameters for Layer Normalization.

    Learnable scale (gamma / weight) is initialized to ones(d_model).
    Learnable bias (beta / bias) is initialized to zeros(d_model).

    Args:
        d_model: Dimensionality of the model embedding / features.
        eps: Epsilon constant for numerical stability (retained for API consistency).
        dtype: Numerical data type for parameters (defaults to float32).

    Returns:
        Dictionary PyTree containing:
            - 'scale': ones of shape (d_model,)
            - 'bias': zeros of shape (d_model,)
    """
    return {
        "scale": jnp.ones((d_model,), dtype=dtype),
        "bias": jnp.zeros((d_model,), dtype=dtype),
    }


def layer_norm(
    params: Mapping[str, Any],
    x: Array,
    eps: float = 1e-5,
) -> Array:
    """Apply Layer Normalization over the last feature dimension.

    Formula:
        x_norm = ((x - mean) / sqrt(var + eps)) * scale + bias

    Mean and variance are computed across the last dimension (-1) with keepdims=True.
    Calculations on low-precision inputs (e.g. bfloat16/float16) are performed in float32
    to prevent numerical instability and variance underflow before casting back.

    Args:
        params: Mapping containing learnable scale and optional bias.
            Supports keys 'scale' (with fallbacks to 'weight' or 'gamma')
            and 'bias' (with fallbacks to 'beta' or 'b').
        x: Input tensor of shape (..., d_model).
        eps: Small epsilon constant added to variance for numerical stability. Default 1e-5.

    Returns:
        Normalized tensor of shape (..., d_model) with identical dtype to x.
    """
    # Parameter key resolution with flexible alias support
    scale = params.get("scale")
    if scale is None:
        scale = params.get("weight", params.get("gamma"))
    if scale is None:
        raise KeyError("LayerNorm params must contain 'scale', 'weight', or 'gamma'.")

    bias = params.get("bias", params.get("beta", params.get("b")))

    # Upcast to float32 if input is half-precision for numerical stability
    input_dtype = x.dtype
    if jnp.issubdtype(input_dtype, jnp.floating) and input_dtype not in (jnp.float32, jnp.float64):
        x_f32 = x.astype(jnp.float32)
        mean = jnp.mean(x_f32, axis=-1, keepdims=True)
        var = jnp.var(x_f32, axis=-1, keepdims=True)
        x_norm = ((x_f32 - mean) / jnp.sqrt(var + eps)).astype(input_dtype)
    else:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + eps)

    out = x_norm * scale
    if bias is not None:
        out = out + bias
    return out


class LayerNorm:
    """Object-oriented wrapper for Layer Normalization."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.d_model = d_model
        self.eps = eps
        self.dtype = dtype

    def init(self, rng: Optional[Array] = None) -> LayerNormParams:
        """Initialize parameters."""
        return init_layer_norm(d_model=self.d_model, eps=self.eps, dtype=self.dtype)

    def __call__(
        self,
        params: Mapping[str, Any],
        x: Array,
        eps: Optional[float] = None,
    ) -> Array:
        """Apply layer norm."""
        return layer_norm(
            params=params,
            x=x,
            eps=self.eps if eps is None else eps,
        )


# ---------------------------------------------------------------------------
# MLP / Feed Forward Network (FFN)
# ---------------------------------------------------------------------------

def init_mlp(
    rng: Array,
    d_model: int,
    d_ff: Optional[int] = None,
    bias: bool = True,
    init_std: float = 0.02,
    n_layer: Optional[int] = None,
    residual_scale: Optional[float] = None,
    dtype: jnp.dtype = jnp.float32,
) -> MLPParams:
    """Initialize parameters for GPT-2 MLP / Feed Forward Network.

    Architecture:
        c_fc:   Linear expansion (d_model -> d_ff, default 4 * d_model)
        gelu:   Activation (tanh approximation)
        c_proj: Linear projection back (d_ff -> d_model)

    Weights are drawn from N(0, init_std^2).
    In GPT-2, weights of residual projections (c_proj) are scaled by
    1 / sqrt(2 * n_layer) to prevent activation variance growth across layers.

    Args:
        rng: JAX PRNGKey for stochastic weight initialization.
        d_model: Dimensionality of model embeddings (in_features and out_features).
        d_ff: Inner expansion dimensionality. Defaults to 4 * d_model.
        bias: Whether to include learnable bias vectors in linear projections. Default True.
        init_std: Base standard deviation for weight initialization. Default 0.02.
        n_layer: Number of transformer layers in the model. If provided, c_proj weights
            are scaled by 1 / sqrt(2 * n_layer) as specified in GPT-2.
        residual_scale: Explicit scaling factor for c_proj weights. Overrides n_layer if given.
        dtype: Numerical data type for parameters (defaults to float32).

    Returns:
        Dictionary PyTree with 'c_fc' and 'c_proj' containing 'w' and 'bias'.
    """
    if d_ff is None:
        d_ff = 4 * d_model

    k_fc, k_proj = jax.random.split(rng, 2)

    # Base standard deviation for c_fc
    w_fc = jax.random.normal(k_fc, shape=(d_model, d_ff), dtype=dtype) * init_std

    # Scaled standard deviation for c_proj on the residual stream
    proj_std = init_std
    if residual_scale is not None:
        proj_std = proj_std * residual_scale
    elif n_layer is not None and n_layer > 0:
        proj_std = proj_std / math.sqrt(2.0 * float(n_layer))

    w_proj = jax.random.normal(k_proj, shape=(d_ff, d_model), dtype=dtype) * proj_std

    params: dict[str, Any] = {
        "c_fc": {"w": w_fc},
        "c_proj": {"w": w_proj},
    }

    if bias:
        params["c_fc"]["bias"] = jnp.zeros((d_ff,), dtype=dtype)
        params["c_proj"]["bias"] = jnp.zeros((d_model,), dtype=dtype)

    return params


def mlp(
    params: Mapping[str, Any],
    x: Array,
    approximate: bool = True,
) -> Array:
    """Apply GPT-2 MLP / Feed Forward Network forward pass.

    Computation:
        h = gelu(x @ c_fc.w + c_fc.bias)
        out = h @ c_proj.w + c_proj.bias

    Args:
        params: Mapping containing 'c_fc' and 'c_proj' parameter dictionaries.
            Each sub-dictionary contains weight 'w' (or 'weight') and optional
            bias 'bias' (or 'b').
        x: Input tensor of shape (..., d_model).
        approximate: Whether to use the GPT-2 tanh GELU approximation. Default True.

    Returns:
        Output tensor of shape (..., d_model).
    """
    # 1. c_fc: Linear expansion (d_model -> 4 * d_model)
    fc_params = params["c_fc"]
    w_fc = fc_params.get("w")
    if w_fc is None:
        w_fc = fc_params["weight"]
    b_fc = fc_params.get("bias", fc_params.get("b"))

    h = jnp.matmul(x, w_fc)
    if b_fc is not None:
        h = h + b_fc

    # 2. GELU activation
    h = gelu(h, approximate=approximate)

    # 3. c_proj: Linear projection back (4 * d_model -> d_model)
    proj_params = params["c_proj"]
    w_proj = proj_params.get("w")
    if w_proj is None:
        w_proj = proj_params["weight"]
    b_proj = proj_params.get("bias", proj_params.get("b"))

    out = jnp.matmul(h, w_proj)
    if b_proj is not None:
        out = out + b_proj

    return out


class MLP:
    """Object-oriented wrapper for GPT-2 MLP / Feed Forward Network."""

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        bias: bool = True,
        initializer_std: float = 0.02,
        n_layer: Optional[int] = None,
        proj_scale: Optional[float] = None,
        approximate: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.bias = bias
        self.initializer_std = initializer_std
        self.n_layer = n_layer
        self.proj_scale = proj_scale
        self.approximate = approximate
        self.dtype = dtype

    def init(self, rng: Array) -> MLPParams:
        """Initialize parameters."""
        return init_mlp(
            rng=rng,
            d_model=self.d_model,
            d_ff=self.d_ff,
            bias=self.bias,
            init_std=self.initializer_std,
            n_layer=self.n_layer,
            residual_scale=self.proj_scale,
            dtype=self.dtype,
        )

    def __call__(
        self,
        params: Mapping[str, Any],
        x: Array,
        approximate: Optional[bool] = None,
    ) -> Array:
        """Apply MLP forward pass."""
        return mlp(
            params=params,
            x=x,
            approximate=self.approximate if approximate is None else approximate,
        )


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------
# In GPT-2 architectures, MLP and Feed Forward Network (FFN) refer to the exact same component.
init_feed_forward = init_mlp
feed_forward = mlp
FeedForward = MLP
