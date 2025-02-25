import torch.nn


class CrossAttention(torch.nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self._mha = torch.nn.MultiheadAttention(
            embedding_size, num_heads, batch_first=True, dropout=dropout_rate
        )
        self._layer_norm = torch.nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor, attn_mask: torch.Tensor):
        x_input = x
        x, _ = self._mha(query=x, key=context, value=context, attn_mask=attn_mask)
        x = torch.add(x, x_input)
        return self._layer_norm(x)


class GlobalSelfAttention(torch.nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self._mha = torch.nn.MultiheadAttention(
            embedding_size, num_heads, batch_first=True, dropout=dropout_rate
        )
        self._layer_norm = torch.nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        x_input = x
        x, _ = self._mha(query=x, key=x, value=x, attn_mask=attn_mask)
        x = torch.add(x, x_input)
        return self._layer_norm(x)


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self._mha = torch.nn.MultiheadAttention(
            embedding_size, num_heads, batch_first=True, dropout=dropout_rate
        )
        self._layer_norm = torch.nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor):
        attn_mask = (
            torch.nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            + padding_mask
        )
        x_input = x
        x, _ = self._mha(query=x, key=x, value=x, attn_mask=attn_mask)
        x = torch.add(x, x_input)
        return self._layer_norm(x)
