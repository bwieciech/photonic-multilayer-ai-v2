import math
from typing import Optional

import torch
from einops import einops
from torch import nn, Tensor


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_instances: Optional[int] = None,
        init_range: float = 0.02,
    ):
        super().__init__()
        assert (
            embed_dim % n_heads == 0
        ), "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_instances = n_instances
        self.head_dim = head_dim = embed_dim // n_heads

        if n_instances is None:
            self.W_Q = nn.Parameter(torch.empty((n_heads, embed_dim, head_dim)))
            self.W_K = nn.Parameter(torch.empty((n_heads, embed_dim, head_dim)))
            self.W_V = nn.Parameter(torch.empty((n_heads, embed_dim, head_dim)))
            self.W_O = nn.Parameter(torch.empty((n_heads, head_dim, embed_dim)))
            self.b_Q = nn.Parameter(torch.zeros((n_heads, head_dim)))
            self.b_K = nn.Parameter(torch.zeros((n_heads, head_dim)))
            self.b_V = nn.Parameter(torch.zeros((n_heads, head_dim)))
            self.b_O = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.W_Q = nn.Parameter(
                torch.empty((n_instances, n_heads, embed_dim, head_dim))
            )
            self.W_K = nn.Parameter(
                torch.empty((n_instances, n_heads, embed_dim, head_dim))
            )
            self.W_V = nn.Parameter(
                torch.empty((n_instances, n_heads, embed_dim, head_dim))
            )
            self.W_O = nn.Parameter(
                torch.empty((n_instances, n_heads, head_dim, embed_dim))
            )
            self.b_Q = nn.Parameter(torch.zeros((n_instances, n_heads, head_dim)))
            self.b_K = nn.Parameter(torch.zeros((n_instances, n_heads, head_dim)))
            self.b_V = nn.Parameter(torch.zeros((n_instances, n_heads, head_dim)))
            self.b_O = nn.Parameter(torch.zeros(n_instances, embed_dim))
        nn.init.normal_(self.W_Q, std=init_range)
        nn.init.normal_(self.W_K, std=init_range)
        nn.init.normal_(self.W_V, std=init_range)
        nn.init.normal_(self.W_O, std=init_range)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        Q = (
            einops.einsum(
                query,
                self.W_Q,
                "batch pos embed_dim, n_heads embed_dim head_dim -> batch pos n_heads head_dim"
                if self.n_instances is None
                else "batch n_instances pos embed_dim, n_instances n_heads embed_dim head_dim -> batch pos n_instances n_heads head_dim",
            )
            + self.b_Q
        )
        K = (
            einops.einsum(
                key,
                self.W_K,
                "batch pos embed_dim, n_heads embed_dim head_dim -> batch pos n_heads head_dim"
                if self.n_instances is None
                else "batch n_instances pos embed_dim, n_instances n_heads embed_dim head_dim -> batch pos n_instances n_heads head_dim",
            )
            + self.b_K
        )
        V = (
            einops.einsum(
                value,
                self.W_V,
                "batch pos embed_dim, n_heads embed_dim head_dim -> batch pos n_heads head_dim"
                if self.n_instances is None
                else "batch n_instances pos embed_dim, n_instances n_heads embed_dim head_dim -> batch pos n_instances n_heads head_dim",
            )
            + self.b_V
        )

        QKT = einops.einsum(
            Q,
            K,
            "batch pos_from n_heads head_dim, batch pos_to n_heads head_dim -> batch n_heads pos_from pos_to"
            if self.n_instances is None
            else "batch pos_from n_instances n_heads head_dim, batch pos_to n_instances n_heads head_dim -> batch n_instances n_heads pos_from pos_to",
        ) / math.sqrt(self.head_dim)

        O = torch.softmax(
            QKT + (torch.zeros_like(QKT) if attn_mask is None else attn_mask), dim=-1
        )

        OV = einops.einsum(
            O,
            V,
            "batch n_heads pos_from pos_to, batch pos_to n_heads head_dim -> batch n_heads pos_from head_dim"
            if self.n_instances is None
            else "batch n_instances n_heads pos_from pos_to, batch pos_to n_instances n_heads head_dim -> batch n_instances n_heads pos_from head_dim",
        )

        result = (
            einops.einsum(
                OV,
                self.W_O,
                "batch n_heads pos head_dim, n_heads head_dim embed_dim -> batch pos embed_dim"
                if self.n_instances is None
                else "batch n_instances n_heads pos head_dim, n_instances n_heads head_dim embed_dim -> batch pos n_instances embed_dim",
            )
            + self.b_O
        )
        return (
            einops.rearrange(
                result,
                "batch pos n_instances embed_dim -> batch n_instances pos embed_dim",
            )
            if self.n_instances is not None
            else result
        )
