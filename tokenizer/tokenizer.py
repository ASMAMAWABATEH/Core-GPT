from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .vocab import Vocabulary


@dataclass
class CharTokenizer:
    """Character-level tokenizer with a fixed vocabulary."""

    vocab: Vocabulary

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        vocab = Vocabulary.build(text)
        return cls(vocab=vocab)

    def encode(self, text: str) -> List[int]:
        return self.vocab.encode(text)

    def decode(self, ids: Iterable[int]) -> str:
        return self.vocab.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.vocab.size
