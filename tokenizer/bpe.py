from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .vocab import Vocabulary


def _count_pairs(tokens: List[str]) -> Counter[Tuple[str, str]]:
    return Counter(zip(tokens, tokens[1:]))


def _merge_pair(tokens: List[str], pair: Tuple[str, str]) -> List[str]:
    merged: List[str] = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            merged.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


@dataclass
class BPETokenizer:
    """Simple BPE tokenizer operating on character sequences."""

    vocab: Vocabulary
    merges: List[Tuple[str, str]]
    unk_token: str = "<unk>"

    @classmethod
    def from_text(cls, text: str, num_merges: int = 1000) -> "BPETokenizer":
        tokens = list(text)
        merges: List[Tuple[str, str]] = []

        for _ in range(num_merges):
            pair_counts = _count_pairs(tokens)
            if not pair_counts:
                break
            (pair_a, pair_b), count = pair_counts.most_common(1)[0]
            if count < 2:
                break
            merges.append((pair_a, pair_b))
            tokens = _merge_pair(tokens, (pair_a, pair_b))

        base_symbols = set(text)
        vocab_tokens = set(base_symbols)
        vocab_tokens.add(cls.unk_token)
        for a, b in merges:
            vocab_tokens.add(a + b)
        vocab = Vocabulary.build(vocab_tokens, add_unk=True, unk_token=cls.unk_token)
        return cls(vocab=vocab, merges=merges, unk_token=cls.unk_token)

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        for pair in self.merges:
            tokens = _merge_pair(tokens, pair)
        return tokens

    def encode(self, text: str) -> List[int]:
        tokens = [
            ch if ch in self.vocab.stoi else self.unk_token
            for ch in text
        ]
        tokens = self._apply_merges(tokens)
        return self.vocab.encode(tokens)

    def decode(self, ids: Iterable[int]) -> str:
        return self.vocab.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.vocab.size
