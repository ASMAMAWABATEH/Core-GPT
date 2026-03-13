from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    """Position-wise feedforward network."""

    def __init__(self, embedding_dim: int, dropout: float, use_bias: bool = True) -> None:
        super().__init__()
        hidden_dim = 4 * embedding_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim, bias=use_bias),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim, bias=use_bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
