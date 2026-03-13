from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch

from scratchgpt.tokenizer.tokenizer import CharTokenizer
from scratchgpt.tokenizer.vocab import Vocabulary


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return path.read_text(encoding="utf-8")


def build_and_save_dataset(raw_path: Path, output_path: Path) -> Dict[str, object]:
    """Tokenize raw text and save processed tensor + vocab."""
    text = load_text(raw_path)
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": data,
        "vocab": {
            "itos": tokenizer.vocab.itos,
            "stoi": tokenizer.vocab.stoi,
        },
    }
    torch.save(payload, output_path)
    return payload


def load_processed_dataset(path: Path) -> Tuple[torch.Tensor, Vocabulary]:
    payload = torch.load(path, map_location="cpu")
    vocab = Vocabulary(stoi=payload["vocab"]["stoi"], itos=payload["vocab"]["itos"])
    data = payload["data"]
    return data, vocab
