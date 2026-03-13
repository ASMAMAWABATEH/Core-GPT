from __future__ import annotations

import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings."""

    def __init__(self, context_length: int, embedding_dim: int) -> None:
        super().__init__()
        self.context_length = context_length
        self.embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device)
        return self.embedding(positions).unsqueeze(0).expand(bsz, seq_len, -1)
