from typing import Optional

import einops
import torch
from torch import nn, Tensor


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_instances: Optional[int] = None,
        init_range: float = 0.02,
    ):
        super().__init__()
        self.n_instances = n_instances
        self.in_features = in_features
        self.out_features = out_features
        if n_instances is None:
            self.W = nn.Parameter(torch.empty((in_features, out_features)))
            self.b = nn.Parameter(torch.zeros(out_features))
        else:
            self.W = nn.Parameter(torch.empty((n_instances, in_features, out_features)))
            self.b = nn.Parameter(torch.zeros((n_instances, out_features)))
        nn.init.normal_(self.W, std=init_range)

    def forward(self, x: Tensor) -> Tensor:
        if self.n_instances is not None and x.ndim == 4:
            x = einops.rearrange(
                x,
                "batch n_instances pos in_features -> batch pos n_instances in_features",
            )
        x = (
            einops.einsum(
                x,
                self.W,
                "... in_features, in_features out_features -> ... out_features"
                if self.n_instances is None
                else "... n_instances in_features, n_instances in_features out_features -> ... n_instances out_features",
            )
            + self.b
        )
        return (
            einops.rearrange(
                x,
                "batch pos n_instances out_features -> batch n_instances pos out_features",
            )
            if self.n_instances is not None and x.ndim == 4
            else x
        )
