from typing import Optional

import torch.nn

from multilayerai.activation import get_activation_instance
from multilayerai.model.layers import Linear, LayerNorm


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        input_embedding_size: int,
        output_embedding_size: int,
        ff_hidden_dim: int,
        activation: str,
        use_layer_norm: bool,
        n_instances: Optional[int] = None,
    ):
        super().__init__()
        self.fc1 = Linear(input_embedding_size, ff_hidden_dim, n_instances)
        self.activation1 = get_activation_instance(
            activation, in_features=self.fc1.out_features
        )
        self.fc2 = Linear(ff_hidden_dim, output_embedding_size, n_instances)
        self.activation2 = get_activation_instance(
            activation, in_features=self.fc2.out_features
        )
        self.layer_norm = (
            LayerNorm(output_embedding_size, n_instances)
            if use_layer_norm
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_input = x
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.add(x, x_input)
        x = self.layer_norm(x)
        return x
