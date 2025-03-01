from torch import nn
from torch.nn import Tanh, Sigmoid, ReLU

from multilayerai.activation.sine import Sine

ACTIVATIONS_BY_NAME = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "sine": Sine,
}


def get_activation_instance(activation_name: str, **kwargs) -> nn.Module:
    if activation_name not in ACTIVATIONS_BY_NAME:
        raise ValueError(
            f"Unknown activation function: {activation_name}, available activations are: {', '.join(list(ACTIVATIONS_BY_NAME.keys()))}"
        )
    if activation_name == "sine":
        return ACTIVATIONS_BY_NAME[activation_name](in_features=kwargs["in_features"])
    return ACTIVATIONS_BY_NAME[activation_name]()
