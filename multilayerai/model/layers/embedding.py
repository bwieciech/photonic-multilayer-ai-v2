from typing import Optional, Any

import einops
import torch
from torch import nn, Tensor


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        n_instances: Optional[int] = None,
        init_range: float = 0.02,
    ):
        super().__init__()
        self.n_instances = n_instances
        self.num_embeddings = num_embeddings

        shape = (
            (num_embeddings, embed_dim)
            if n_instances is None
            else (n_instances, num_embeddings, embed_dim)
        )
        self.W = nn.Parameter(torch.empty(shape))
        nn.init.normal_(self.W, std=init_range)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.nn.functional.one_hot(x, num_classes=self.num_embeddings).float()
        return einops.einsum(
            x,
            self.W,
            "batch pos num_embeddings, num_embeddings embed_dim -> batch pos embed_dim"
            if self.n_instances is None
            else "batch n_instances pos num_embeddings, n_instances num_embeddings embed_dim -> batch n_instances pos embed_dim",
        )

    def __getitem__(self, item: Any) -> Tensor:
        return self.W[item]
