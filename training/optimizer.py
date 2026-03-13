from __future__ import annotations

from typing import Iterable

import torch


def build_optimizer(params: Iterable[torch.nn.Parameter], learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
