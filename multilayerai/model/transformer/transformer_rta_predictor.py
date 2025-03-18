import math
from typing import Any, Callable, Collection, Union, Optional

import einops
import torch
from collections import defaultdict
from torch import nn

from multilayerai.activation import get_activation_instance
from multilayerai.model.layers import Linear
from multilayerai.model.transformer.blocks.encoder import EncoderBlock


class TransformerRTAPredictor(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        embedding_size: int,
        padding_token_idx: int,
        num_heads: int,
        ff_hidden_dim: int,
        num_encoder_blocks: Union[int, Collection[int]],
        num_wavelengths: int,
        max_seq_len: int = 32,
        activation: str = "relu",
        use_layer_norm: bool = True,
        cache: bool = False,
    ):
        super().__init__()

        self.padding_token_idx = padding_token_idx
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.num_wavelengths = num_wavelengths

        self.material_embedder = nn.Embedding(num_tokens, embedding_size)
        self.thickness_encoder = nn.Linear(1, embedding_size)
        self.positional_encodings = self._create_positional_encoding(
            max_seq_len, embedding_size
        )
        self.register_buffer("num_encoder_blocks", torch.tensor(num_encoder_blocks))
        self.n_instances = (
            None if isinstance(num_encoder_blocks, int) else len(num_encoder_blocks)
        )
        self.encoder_blocks = torch.nn.ModuleList(
            [
                EncoderBlock(
                    embedding_size,
                    embedding_size,
                    num_heads,
                    ff_hidden_dim,
                    activation,
                    use_layer_norm,
                    self.n_instances,
                )
                for _ in range(
                    max(num_encoder_blocks)
                    if isinstance(num_encoder_blocks, Collection)
                    else num_encoder_blocks
                )
            ]
        )
        self.output_head = nn.Sequential(
            Linear(embedding_size, ff_hidden_dim, self.n_instances),
            get_activation_instance(activation, in_features=ff_hidden_dim),
            Linear(
                ff_hidden_dim, num_wavelengths * 3, self.n_instances
            ),  # Predict R, T, A for each wavelength
        )
        self.cache = cache
        if cache:
            self.cached_values = defaultdict(dict)
            self._register_hooks()

    def forward(
        self, materials: torch.Tensor, thicknesses: torch.Tensor
    ) -> torch.Tensor:
        if self.cache:
            self.cached_values = defaultdict(dict)

        batch_size, seq_len = materials.shape
        input_padding_arr = torch.where(
            materials == self.padding_token_idx, -torch.inf, 0
        )
        input_padding_mask = self._create_padding_mask(input_padding_arr)

        materials_embeddings = self.material_embedder(materials.int())
        thickness_encoding = self.thickness_encoder(thicknesses.unsqueeze(-1).float())
        positional_encoding = self.positional_encodings[:, :seq_len, :].to(
            materials.device
        )
        x = materials_embeddings + thickness_encoding + positional_encoding
        if self.n_instances is not None:
            x = einops.repeat(
                x,
                "batch pos embedding_size -> batch n_instances pos embedding_size",
                n_instances=self.n_instances,
            )
        for block_number, encoder_block in enumerate(self.encoder_blocks):
            if self.n_instances is None:
                x = encoder_block(x, input_padding_mask)
            else:
                x_in = x
                x = encoder_block(x, input_padding_mask)
                x_copy_in = self.num_encoder_blocks <= block_number
                x[:, x_copy_in] = x_in[:, x_copy_in]

        x = x.sum(dim=-2)
        rta_flat = self.output_head(x)
        output_shape = (
            (batch_size, self.num_wavelengths, 3)
            if self.n_instances is None
            else (batch_size, self.n_instances, self.num_wavelengths, 3)
        )
        x = rta_flat.view(output_shape)

        return torch.softmax(x, dim=-1)

    def _create_positional_encoding(
        self, max_len: int, embedding_size: int
    ) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2).float()
            * (-math.log(10000.0) / embedding_size)
        )
        pe = torch.zeros(max_len, embedding_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: embedding_size // 2])
        return pe.unsqueeze(0)

    def _create_padding_mask(self, arr: torch.Tensor) -> torch.Tensor:
        mask = arr.unsqueeze(1).expand(-1, arr.size(1), -1)
        if self.n_instances is None:
            return einops.repeat(
                mask,
                "batch pos_from pos_to -> batch num_heads pos_from pos_to",
                num_heads=self.num_heads,
            )
        else:
            return einops.repeat(
                mask,
                "batch pos_from pos_to -> batch n_instances num_heads pos_from pos_to",
                n_instances=self.n_instances,
                num_heads=self.num_heads,
            )

    def _register_hooks(self) -> None:
        def get_hook(module_name: str) -> Callable[[torch.nn.Module, Any, Any], None]:
            def hook_fn(
                module: torch.nn.Module, input: torch.Tensor, output: Any
            ) -> None:
                self.cached_values["outputs"][module_name] = output

            return hook_fn

        named_modules = dict(self.named_modules())
        for module_name, module in self.named_modules():
            if not module_name:
                continue
            module.register_forward_hook(get_hook(module_name))
