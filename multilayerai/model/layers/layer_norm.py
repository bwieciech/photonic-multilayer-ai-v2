from typing import Union, Optional

import einops
from torch import nn, Tensor, Size
import torch


class LayerNorm(torch.nn.Module):
    def __init__(
        self,
        in_features: Union[int, Size],
        n_instances: Optional[int] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.n_instances = n_instances
        self.eps = eps
        if n_instances is None:
            self.W = nn.Parameter(torch.ones(in_features))
            self.b = nn.Parameter(torch.zeros(in_features))
        else:
            self.W = nn.Parameter(torch.ones((n_instances, in_features)))
            self.b = nn.Parameter(torch.zeros((n_instances, in_features)))

    def forward(self, x: Tensor) -> Tensor:
        if self.n_instances is not None and x.ndim == 4:
            x = einops.rearrange(
                x,
                "batch n_instances pos in_features -> batch pos n_instances in_features",
            )
        mean = x.mean(dim=-1, keepdim=True)
        std = (x.var(dim=-1, unbiased=False, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        x = einops.einsum(
            x,
            self.W,
            "... in_features, in_features -> ... in_features"
            if self.n_instances is None
            else "... n_instances in_features, n_instances in_features -> ... n_instances in_features",
        )
        return (
            einops.rearrange(
                x,
                "batch pos n_instances out_features -> batch n_instances pos out_features",
            )
            if self.n_instances is not None and x.ndim == 4
            else x
        )