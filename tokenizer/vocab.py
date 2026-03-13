from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class Vocabulary:
    """Simple vocabulary for tokenization."""

    stoi: Dict[str, int]
    itos: List[str]
    unk_token: Optional[str] = None

    @classmethod
    def build(
        cls,
        tokens: Iterable[str],
        add_unk: bool = False,
        unk_token: str = "<unk>",
    ) -> "Vocabulary":
        unique = sorted(set(tokens))
        if add_unk and unk_token not in unique:
            unique = [unk_token] + unique
        itos = list(unique)
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos, unk_token=unk_token if add_unk else None)

    def encode(self, tokens: Iterable[str]) -> List[int]:
        ids: List[int] = []
        for token in tokens:
            if token in self.stoi:
                ids.append(self.stoi[token])
            elif self.unk_token is not None:
                ids.append(self.stoi[self.unk_token])
            else:
                raise KeyError(token)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    @property
    def size(self) -> int:
        return len(self.itos)
