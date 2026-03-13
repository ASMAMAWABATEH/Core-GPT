from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class Vocabulary:
    """Simple vocabulary for character-level tokenization."""

    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(cls, tokens: Iterable[str]) -> "Vocabulary":
        unique = sorted(set(tokens))
        itos = list(unique)
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    @property
    def size(self) -> int:
        return len(self.itos)
