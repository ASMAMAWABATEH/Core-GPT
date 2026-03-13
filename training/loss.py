from __future__ import annotations

import torch
from torch import nn


def language_modeling_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for next-token prediction."""
    return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
