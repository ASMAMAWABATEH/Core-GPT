from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Token embedding layer."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
