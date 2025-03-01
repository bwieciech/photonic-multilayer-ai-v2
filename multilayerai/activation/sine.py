import torch
from torch import nn


class Sine(nn.Module):
    def __init__(self, in_features: int, omega_0: float = 30.0):
        super().__init__()
        self._in_features = in_features
        self._omega = nn.Parameter(torch.full((in_features,), fill_value=omega_0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self._omega * x)

    def extra_repr(self) -> str:
        return f"in_features={self._in_features}"
