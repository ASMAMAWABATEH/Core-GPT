from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class TextDataset(Dataset):
    """Character-level dataset for next-token prediction."""

    data: torch.Tensor
    context_length: int

    def __len__(self) -> int:
        return max(0, self.data.size(0) - self.context_length)

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return x, y
