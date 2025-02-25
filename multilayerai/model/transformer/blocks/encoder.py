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
        dropout_rate: float,
    ):
        super().__init__()
        self._self_attention = GlobalSelfAttention(
            input_embedding_size, num_heads, dropout_rate
        )
        self._feed_forward = FeedForward(
            input_embedding_size, output_embedding_size, ff_hidden_dim
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = self._self_attention(x, attn_mask=padding_mask)
        x = self._feed_forward(x)
        return x
