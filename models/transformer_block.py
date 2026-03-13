from __future__ import annotations

import torch
from torch import nn

from .attention import MultiHeadSelfAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Decoder-only Transformer block with pre-norm."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.attn = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout,
            use_bias=use_bias,
        )
        self.ffn = FeedForward(embedding_dim=embedding_dim, dropout=dropout, use_bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
