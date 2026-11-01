"""Pure JAX implementation of the GPT-2 Transformer Architecture.

This module composes the pure JAX primitives (LayerNorm, CausalSelfAttention, and MLP)
into a complete autoregressive language model.

Features:
- Pure JAX: Compatible with `jax.jit`, `jax.grad`, `jax.vmap`, and JAX PyTree serialization.
- Configurable architectures: Pre-tuned configurations for 53M (L=2) and 68M (L=4) GPT-2.
- Weight Tying: Option to tie output projection (LM head) weights with token embeddings (wte).
- Numerical stability: Built-in cross-entropy loss computation using log-softmax.
- Text Generation: Autoregressive token generation supporting greedy and temperature/top-k sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

from gpt.attention import (
    causal_self_attention,
    init_causal_self_attention,
)
from gpt.layers import (
    init_layer_norm,
    init_mlp,
    layer_norm,
    mlp,
)


# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPTConfig:
    """Hyperparameters for GPT-2 architecture."""

    vocab_size: int = 50257        # GPT-2 tokenizer vocabulary size
    max_seq_len: int = 1024       # Maximum context length / sequence length
    d_model: int = 768            # Hidden dimension / embedding size
    n_layers: int = 12            # Number of transformer blocks
    n_heads: int = 12             # Number of attention heads (d_model must be divisible by n_heads)
    d_ff: Optional[int] = None    # MLP intermediate dimension (defaults to 4 * d_model)
    bias: bool = True             # Whether linear projections and LayerNorm use bias
    tie_word_embeddings: bool = True  # Whether to tie wte weights to lm_head
    dropout_rate: float = 0.0     # Dropout probability
    initializer_std: float = 0.02 # Base standard deviation for weight initialization
    layer_norm_eps: float = 1e-5  # Epsilon for LayerNorm numerical stability

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

    @property
    def intermediate_dim(self) -> int:
        return self.d_ff if self.d_ff is not None else 4 * self.d_model

    @classmethod
    def gpt2_53m(cls, max_seq_len: int = 1024) -> "GPTConfig":
        """Configuration for ~53.6M parameter GPT-2 variant.

        Uses canonical GPT-2 width (d_model=768, n_heads=12) with L=2 layers.
        Closest fit to ~50M target while preserving full 768-dim GPT-2 width:
          - Body: 14.18M parameters
          - Embeddings (We + Wp): 39.38M parameters
          - Total: 53,561,088 (~53.6M) parameters with tied embeddings.
        """
        return cls(
            vocab_size=50257,
            max_seq_len=max_seq_len,
            d_model=768,
            n_layers=2,
            n_heads=12,
            dropout_rate=0.0,
        )

    @classmethod
    def gpt2_68m(cls, max_seq_len: int = 1024) -> "GPTConfig":
        """Configuration for ~67.7M parameter mini-GPT baseline.

        Uses canonical GPT-2 width (d_model=768, n_heads=12) with L=4 layers.
        Closest fit to ~70M target while preserving full 768-dim GPT-2 width:
          - Body: 28.35M parameters
          - Embeddings (We + Wp): 39.38M parameters
          - Total: 67,736,832 (~67.7M) parameters with tied embeddings.
        """
        return cls(
            vocab_size=50257,
            max_seq_len=max_seq_len,
            d_model=768,
            n_layers=4,
            n_heads=12,
            dropout_rate=0.0,
        )


# ---------------------------------------------------------------------------
# Parameter Initialization
# ---------------------------------------------------------------------------

def init_gpt2(
    rng: Array,
    config: GPTConfig,
    dtype: jnp.dtype = jnp.float32,
) -> Dict[str, Any]:
    """Initializes all parameters for the GPT-2 model.

    Parameter structure:
        - transformer:
            - wte: Token embedding table (vocab_size, d_model)
            - wpe: Positional embedding table (max_seq_len, d_model)
            - h: List of n_layers transformer blocks:
                - ln_1: LayerNorm (scale, bias)
                - attn: CausalSelfAttention (c_attn, c_proj)
                - ln_2: LayerNorm (scale, bias)
                - mlp:  MLP (c_fc, c_proj)
            - ln_f: Final LayerNorm (scale, bias)
        - lm_head: Output projection to vocab_size (if not tied)

    Args:
        rng: JAX PRNGKey.
        config: GPTConfig instance.
        dtype: Numerical data type for parameters (defaults to float32).

    Returns:
        Nested dictionary of JAX arrays (PyTree).
    """
    rng, k_wte, k_wpe = jax.random.split(rng, 3)

    # 1. Word Token Embeddings (wte)
    wte = jax.random.normal(
        k_wte,
        shape=(config.vocab_size, config.d_model),
        dtype=dtype,
    ) * config.initializer_std

    # 2. Positional Embeddings (wpe)
    wpe = jax.random.normal(
        k_wpe,
        shape=(config.max_seq_len, config.d_model),
        dtype=dtype,
    ) * (config.initializer_std * 0.5)

    # 3. Transformer Blocks (h)
    blocks = []
    for layer_idx in range(config.n_layers):
        rng, k_attn, k_mlp = jax.random.split(rng, 3)
        block = {
            "ln_1": init_layer_norm(config.d_model, eps=config.layer_norm_eps, dtype=dtype),
            "attn": init_causal_self_attention(
                rng=k_attn,
                d_model=config.d_model,
                n_heads=config.n_heads,
                max_seq_len=config.max_seq_len,
                bias=config.bias,
                initializer_std=config.initializer_std,
                n_layer=config.n_layers,
                dtype=dtype,
            ),
            "ln_2": init_layer_norm(config.d_model, eps=config.layer_norm_eps, dtype=dtype),
            "mlp": init_mlp(
                rng=k_mlp,
                d_model=config.d_model,
                d_ff=config.intermediate_dim,
                bias=config.bias,
                init_std=config.initializer_std,
                n_layer=config.n_layers,
                dtype=dtype,
            ),
        }
        blocks.append(block)

    # 4. Final LayerNorm (ln_f)
    ln_f = init_layer_norm(config.d_model, eps=config.layer_norm_eps, dtype=dtype)

    params: Dict[str, Any] = {
        "transformer": {
            "wte": wte,
            "wpe": wpe,
            "h": blocks,
            "ln_f": ln_f,
        }
    }

    # 5. LM Head (if not tied)
    if not config.tie_word_embeddings:
        rng, k_lm = jax.random.split(rng)
        w_lm = jax.random.normal(
            k_lm,
            shape=(config.d_model, config.vocab_size),
            dtype=dtype,
        ) * config.initializer_std
        params["lm_head"] = {"w": w_lm}
        if config.bias:
            params["lm_head"]["b"] = jnp.zeros((config.vocab_size,), dtype=dtype)

    return params


def count_parameters(params: Mapping[str, Any]) -> int:
    """Computes total number of scalar parameters in the model PyTree."""
    leaves = jax.tree_util.tree_leaves(params)
    return sum(x.size for x in leaves)


# ---------------------------------------------------------------------------
# Forward Pass
# ---------------------------------------------------------------------------

def gpt2_forward(
    params: Mapping[str, Any],
    token_ids: Array,
    config: GPTConfig,
    deterministic: bool = True,
    rng: Optional[Array] = None,
) -> Array:
    """Computes forward pass of GPT-2 and returns logits over vocabulary.

    Args:
        params: Model parameter dictionary PyTree.
        token_ids: Integer token indices of shape (batch_size, seq_len).
        config: GPTConfig model configuration.
        deterministic: If True, dropout is disabled (inference mode).
        rng: PRNGKey required if deterministic=False and config.dropout_rate > 0.

    Returns:
        Logits tensor of shape (batch_size, seq_len, vocab_size).
    """
    batch_size, seq_len = token_ids.shape
    if seq_len > config.max_seq_len:
        raise ValueError(
            f"Sequence length {seq_len} exceeds maximum sequence length {config.max_seq_len}"
        )

    t_params = params["transformer"]

    # 1. Word token embeddings: (B, T) -> (B, T, d_model)
    tok_emb = t_params["wte"][token_ids]

    # 2. Learned position embeddings: (T,) -> (1, T, d_model)
    positions = jnp.arange(0, seq_len, dtype=jnp.int32)
    pos_emb = t_params["wpe"][positions]
    pos_emb = jnp.expand_dims(pos_emb, axis=0)

    # 3. Combine embeddings
    x = tok_emb + pos_emb

    # Optional embedding dropout
    if (not deterministic) and (config.dropout_rate > 0.0):
        if rng is None:
            raise ValueError("rng is required for stochastic forward pass with dropout_rate > 0")
        rng, k_drop = jax.random.split(rng)
        keep_prob = 1.0 - config.dropout_rate
        mask = jax.random.bernoulli(k_drop, p=keep_prob, shape=x.shape)
        x = jnp.where(mask, x / keep_prob, 0.0)

    # 4. Transformer Blocks
    blocks = t_params["h"]
    for block in blocks:
        # Dropout key for attention if needed
        k_attn = None
        if (not deterministic) and (config.dropout_rate > 0.0):
            rng, k_attn = jax.random.split(rng)

        # Pre-LN Self-Attention with residual connection:
        # x = x + Attention(LN_1(x))
        x_norm1 = layer_norm(block["ln_1"], x, eps=config.layer_norm_eps)
        attn_out = causal_self_attention(
            params=block["attn"],
            x=x_norm1,
            n_heads=config.n_heads,
            deterministic=deterministic,
            rng=k_attn,
            dropout_rate=config.dropout_rate,
        )
        x = x + attn_out

        # Pre-LN MLP with residual connection:
        # x = x + MLP(LN_2(x))
        x_norm2 = layer_norm(block["ln_2"], x, eps=config.layer_norm_eps)
        mlp_out = mlp(block["mlp"], x_norm2, approximate=True)
        x = x + mlp_out

    # 5. Final LayerNorm
    x = layer_norm(t_params["ln_f"], x, eps=config.layer_norm_eps)

    # 6. LM Head (Decoding Projection to Logits)
    if config.tie_word_embeddings:
        # Tied weights: logits = x @ wte^T
        logits = jnp.matmul(x, t_params["wte"].T)
    else:
        lm_head = params["lm_head"]
        w = lm_head.get("w", lm_head.get("weight"))
        logits = jnp.matmul(x, w)
        b = lm_head.get("b", lm_head.get("bias"))
        if b is not None:
            logits = logits + b

    return logits


# ---------------------------------------------------------------------------
# Loss Computation
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits: Array, targets: Array) -> Array:
    """Computes numerically stable cross-entropy loss for next-token prediction.

    Args:
        logits: Logits of shape (..., vocab_size).
        targets: Ground truth integer token IDs of shape (...).

    Returns:
        Scalar mean cross-entropy loss.
    """
    # log_softmax across vocabulary dimension
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Gather log-probabilities corresponding to true target tokens
    # One-hot indexing without allocating full one-hot tensor:
    # log_p[..., target]
    target_log_probs = jnp.take_along_axis(
        log_probs,
        jnp.expand_dims(targets, axis=-1),
        axis=-1,
    )
    return -jnp.mean(target_log_probs)


def gpt2_loss(
    params: Mapping[str, Any],
    inputs: Array,
    targets: Array,
    config: GPTConfig,
    deterministic: bool = True,
    rng: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Computes both scalar loss and logits for a training/eval batch.

    Args:
        params: Model parameter dictionary.
        inputs: Input token IDs of shape (batch_size, seq_len).
        targets: Target token IDs of shape (batch_size, seq_len).
        config: GPTConfig instance.
        deterministic: If True, dropout is disabled.
        rng: PRNGKey for dropout if stochastic.

    Returns:
        (loss, logits) tuple where loss is a scalar and logits has shape (B, T, V).
    """
    logits = gpt2_forward(
        params=params,
        token_ids=inputs,
        config=config,
        deterministic=deterministic,
        rng=rng,
    )
    loss = cross_entropy_loss(logits, targets)
    return loss, logits


# ---------------------------------------------------------------------------
# Autoregressive Generation
# ---------------------------------------------------------------------------

def generate(
    params: Mapping[str, Any],
    prompt_tokens: Array,
    config: GPTConfig,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng: Optional[Array] = None,
    eot_token: Optional[int] = 50256,
) -> Array:
    """Autoregressively generates new tokens given a prompt.

    Args:
        params: Model parameter PyTree.
        prompt_tokens: Initial prompt token IDs of shape (batch_size, seq_len).
        config: Model configuration.
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature (1.0 = standard, lower = more greedy, 0.0 = pure argmax).
        top_k: If provided, restricts sampling to the top-k highest probability tokens.
        rng: JAX PRNGKey for sampling (required if temperature > 0.0).
        eot_token: End-of-text token ID. If all sequences generate EOT, generation may stop.

    Returns:
        Extended token IDs array of shape (batch_size, seq_len + new_tokens).
    """
    tokens = prompt_tokens

    for _ in range(max_new_tokens):
        # Crop context window if sequence length exceeds max_seq_len
        curr_tokens = tokens if tokens.shape[1] <= config.max_seq_len else tokens[:, -config.max_seq_len:]

        # Forward pass (always deterministic during generation)
        logits = gpt2_forward(
            params=params,
            token_ids=curr_tokens,
            config=config,
            deterministic=True,
        )

        # Focus on the logits for the last token position: (batch_size, vocab_size)
        next_token_logits = logits[:, -1, :]

        if temperature <= 0.0 or rng is None:
            # Greedy argmax selection
            next_token = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
        else:
            rng, k_sample = jax.random.split(rng)
            scaled_logits = next_token_logits / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                # Top-K filtering: mask out all logits outside the top-k
                top_k = min(top_k, scaled_logits.shape[-1])
                top_values, _ = jax.lax.top_k(scaled_logits, k=top_k)
                min_threshold = top_values[:, -1:]
                scaled_logits = jnp.where(
                    scaled_logits < min_threshold,
                    -1e9,
                    scaled_logits,
                )

            # Categorical sampling from softmax distribution
            next_token = jax.random.categorical(k_sample, scaled_logits, axis=-1)
            next_token = jnp.expand_dims(next_token, axis=-1)

        tokens = jnp.concatenate([tokens, next_token], axis=-1)

    return tokens


# ---------------------------------------------------------------------------
# Object-Oriented GPT2 Model Wrapper
# ---------------------------------------------------------------------------

class GPT2:
    """Object-oriented wrapper around pure JAX GPT-2 functions.

    Maintains parameter statelessness while providing a clean, intuitive API.
    """

    def __init__(self, config: GPTConfig):
        self.config = config

    def init(self, rng: Array, dtype: jnp.dtype = jnp.float32) -> Dict[str, Any]:
        """Initializes model parameters."""
        return init_gpt2(rng=rng, config=self.config, dtype=dtype)

    def __call__(
        self,
        params: Mapping[str, Any],
        token_ids: Array,
        deterministic: bool = True,
        rng: Optional[Array] = None,
    ) -> Array:
        """Runs forward pass and returns logits."""
        return gpt2_forward(
            params=params,
            token_ids=token_ids,
            config=self.config,
            deterministic=deterministic,
            rng=rng,
        )

    def loss(
        self,
        params: Mapping[str, Any],
        inputs: Array,
        targets: Array,
        deterministic: bool = True,
        rng: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Computes next-token prediction cross-entropy loss."""
        return gpt2_loss(
            params=params,
            inputs=inputs,
            targets=targets,
            config=self.config,
            deterministic=deterministic,
            rng=rng,
        )

    def generate(
        self,
        params: Mapping[str, Any],
        prompt_tokens: Array,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        rng: Optional[Array] = None,
    ) -> Array:
        """Autoregressively generates text tokens."""
        return generate(
            params=params,
            prompt_tokens=prompt_tokens,
            config=self.config,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )

    def count_parameters(self, params: Mapping[str, Any]) -> int:
        """Counts total trainable parameters."""
        return count_parameters(params)

