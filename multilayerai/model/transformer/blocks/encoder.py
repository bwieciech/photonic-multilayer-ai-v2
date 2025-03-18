from typing import Optional

import torch.nn

from multilayerai.model.transformer.blocks.attention import (
    GlobalSelfAttention,
)
from multilayerai.model.transformer.blocks.feed_forward import FeedForward


class EncoderBlock(torch.nn.Module):
    def __init__(
        self,
        input_embedding_size: int,
        output_embedding_size: int,
        num_heads: int,
        ff_hidden_dim: int,
        activation: str,
        use_layer_norm: bool,
        n_instances: Optional[int] = None,
    ):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            input_embedding_size, num_heads, n_instances
        )
        self.feed_forward = FeedForward(
            input_embedding_size,
            output_embedding_size,
            ff_hidden_dim,
            activation,
            use_layer_norm,
            n_instances,
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.self_attention(x, attn_mask=padding_mask)
        x = self.feed_forward(x)
        return x
