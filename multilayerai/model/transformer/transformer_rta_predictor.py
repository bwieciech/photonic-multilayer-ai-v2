import math
from typing import Any, Callable

import torch
from collections import defaultdict
from torch import nn

from multilayerai.activation import get_activation_instance
from multilayerai.model.transformer.blocks.encoder import EncoderBlock


class TransformerRTAPredictor(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        embedding_size: int,
        padding_token_idx: int,
        num_heads: int,
        ff_hidden_dim: int,
        num_encoder_blocks: int,
        num_wavelengths: int,
        dropout_rate: float = 0.1,
        max_seq_len: int = 32,
        activation: str = "relu",
        use_layer_norm: bool = True,
        cache: bool = False,
    ):
        super().__init__()

        self._padding_token_idx = padding_token_idx
        self._num_heads = num_heads
        self._embedding_size = embedding_size
        self._num_wavelengths = num_wavelengths

        self._material_embedder = nn.Embedding(num_tokens, embedding_size)
        self._thickness_encoder = nn.Linear(1, embedding_size)
        self._positional_encodings = self._create_positional_encoding(
            max_seq_len, embedding_size
        )
        self.encoder_blocks = torch.nn.ModuleList(
            [
                EncoderBlock(
                    embedding_size,
                    embedding_size,
                    num_heads,
                    ff_hidden_dim,
                    dropout_rate,
                    activation,
                    use_layer_norm,
                )
                for _ in range(num_encoder_blocks)
            ]
        )
        self._output_head = nn.Sequential(
            nn.Linear(embedding_size, ff_hidden_dim),
            get_activation_instance(activation, in_features=ff_hidden_dim),
            nn.Linear(
                ff_hidden_dim, num_wavelengths * 3
            ),  # Predict R, T, A for each wavelength
        )
        self._dropout = torch.nn.Dropout(dropout_rate)
        self._cache = cache
        if cache:
            self._cached_values = defaultdict(dict)
            self._register_hooks()

    def forward(
        self, materials: torch.Tensor, thicknesses: torch.Tensor
    ) -> torch.Tensor:
        if self._cache:
            self._cached_values = defaultdict(dict)

        batch_size, seq_len = materials.shape
        input_padding_arr = torch.where(
            materials == self._padding_token_idx, -torch.inf, 0
        )
        input_padding_mask = self._create_padding_mask(input_padding_arr)

        materials_embeddings = self._material_embedder(materials.int())
        thickness_encoding = self._thickness_encoder(thicknesses.unsqueeze(-1).float())
        positional_encoding = self._positional_encodings[:, :seq_len, :].to(
            materials.device
        )
        x = materials_embeddings + thickness_encoding + positional_encoding
        x = self._dropout(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, input_padding_mask)
            x = self._dropout(x)

        x = x.sum(dim=1) / (materials != self._padding_token_idx).sum(dim=-1).unsqueeze(
            dim=-1
        )
        rta_flat = self._output_head(x)
        rta_pred = rta_flat.view(batch_size, self._num_wavelengths, 3)

        rta_pred = torch.softmax(rta_pred, dim=-1)

        return rta_pred

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
        return torch.repeat_interleave(mask, self._num_heads, dim=0)

    def _register_hooks(self) -> None:
        def get_hook(module_name: str) -> Callable[[torch.nn.Module, Any, Any], None]:
            def hook_fn(
                module: torch.nn.Module, input: torch.Tensor, output: Any
            ) -> None:
                self._cached_values["outputs"][module_name] = output

            return hook_fn

        named_modules = dict(self.named_modules())
        for module_name, module in self.named_modules():
            if not module_name:
                continue
            module.register_forward_hook(get_hook(module_name))
