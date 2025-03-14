import math

import torch


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, max_len: int = 128):
        super().__init__()
        self._embedding = torch.nn.Embedding(vocab_size, embedding_size)
        # + 2 for encoding thickness and is_inf thickness flag, appended to the embedding
        self.positional_encoding = PositionalEncoding(max_len, embedding_size + 2)
        self.embedding_size = embedding_size + 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        material = x[:, :, 0].int()
        thickness_info = x[:, :, 1:]
        embedding = self._embedding(material)
        x = torch.cat([embedding, thickness_info], dim=-1)
        x = math.sqrt(self.embedding_size) * x
        return self.positional_encoding(x)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_len: int, embedding_size: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size)
        )
        self._encoding = torch.zeros(max_len, embedding_size).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._encoding[:, 0::2] = torch.sin(position * div_term)
        self._encoding[:, 1::2] = torch.cos(
            position * (div_term if embedding_size % 2 == 0 else div_term[:-1])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._encoding[: x.size(1)]
