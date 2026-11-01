"""JAX-GPT package."""

from gpt.attention import (
    AttentionConfig,
    CausalSelfAttention,
    causal_self_attention,
    init_causal_self_attention,
)
from gpt.layers import (
    FFNParams,
    FeedForward,
    FeedForwardConfig,
    LayerNorm,
    LayerNormConfig,
    LayerNormParams,
    LinearParams,
    MLP,
    MLPConfig,
    MLPParams,
    feed_forward,
    gelu,
    init_feed_forward,
    init_layer_norm,
    init_mlp,
    layer_norm,
    mlp,
)
from gpt.gpt import (
    GPT2,
    GPTConfig,
    count_parameters,
    cross_entropy_loss,
    generate,
    gpt2_forward,
    gpt2_loss,
    init_gpt2,
)

__all__ = [
    # Full Model
    "GPT2",
    "GPTConfig",
    "init_gpt2",
    "gpt2_forward",
    "gpt2_loss",
    "cross_entropy_loss",
    "generate",
    "count_parameters",
    # Attention
    "AttentionConfig",
    "CausalSelfAttention",
    "causal_self_attention",
    "init_causal_self_attention",
    # LayerNorm
    "LayerNorm",
    "LayerNormConfig",
    "LayerNormParams",
    "init_layer_norm",
    "layer_norm",
    # Activation
    "gelu",
    # MLP / FeedForward
    "FFNParams",
    "FeedForward",
    "FeedForwardConfig",
    "LinearParams",
    "MLP",
    "MLPConfig",
    "MLPParams",
    "feed_forward",
    "init_feed_forward",
    "init_mlp",
    "mlp",
]
