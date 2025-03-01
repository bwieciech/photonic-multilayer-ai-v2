import torch.nn

from multilayerai.activation import get_activation_instance


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        input_embedding_size: int,
        output_embedding_size: int,
        ff_hidden_dim: int,
        activation: str,
        use_layer_norm: bool,
    ):
        super().__init__()
        self._fc1 = torch.nn.Linear(input_embedding_size, ff_hidden_dim)
        self._activation1 = get_activation_instance(
            activation, in_features=self._fc1.out_features
        )
        self._fc2 = torch.nn.Linear(ff_hidden_dim, output_embedding_size)
        self._activation2 = get_activation_instance(
            activation, in_features=self._fc2.out_features
        )
        self._layer_norm = (
            torch.nn.LayerNorm(output_embedding_size)
            if use_layer_norm
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_input = x
        x = torch.relu(self._fc1(x))
        x = self._fc2(x)
        x = torch.add(x, x_input)
        x = self._layer_norm(x)
        return x
