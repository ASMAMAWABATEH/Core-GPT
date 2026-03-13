from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch

from tokenizer.bpe import BPETokenizer
from tokenizer.tokenizer import CharTokenizer
from tokenizer.vocab import Vocabulary


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return path.read_text(encoding="utf-8")


def build_and_save_dataset(
    raw_path: Path,
    output_path: Path,
    tokenizer_type: str = "bpe",
    bpe_merges: int = 1000,
) -> Dict[str, object]:
    """Tokenize raw text and save processed tensor + tokenizer."""
    text = load_text(raw_path)
    if tokenizer_type == "char":
        tokenizer = CharTokenizer.from_text(text)
    elif tokenizer_type == "bpe":
        tokenizer = BPETokenizer.from_text(text, num_merges=bpe_merges)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {"data": data}
    if tokenizer_type == "char":
        payload["tokenizer"] = {
            "type": "char",
            "vocab": {
                "itos": tokenizer.vocab.itos,
                "stoi": tokenizer.vocab.stoi,
                "unk_token": tokenizer.vocab.unk_token,
            },
        }
    else:
        payload["tokenizer"] = {
            "type": "bpe",
            "vocab": {
                "itos": tokenizer.vocab.itos,
                "stoi": tokenizer.vocab.stoi,
                "unk_token": tokenizer.vocab.unk_token,
            },
            "merges": tokenizer.merges,
        }
    torch.save(payload, output_path)
    return payload


def load_processed_dataset(path: Path) -> Tuple[torch.Tensor, object]:
    payload = torch.load(path, map_location="cpu")
    data = payload["data"]

    if "tokenizer" not in payload:
        vocab = Vocabulary(
            stoi=payload["vocab"]["stoi"],
            itos=payload["vocab"]["itos"],
        )
        tokenizer = CharTokenizer(vocab=vocab)
        return data, tokenizer

    tok = payload["tokenizer"]
    vocab = Vocabulary(
        stoi=tok["vocab"]["stoi"],
        itos=tok["vocab"]["itos"],
        unk_token=tok["vocab"].get("unk_token"),
    )
    if tok["type"] == "char":
        tokenizer = CharTokenizer(vocab=vocab)
    elif tok["type"] == "bpe":
        tokenizer = BPETokenizer(vocab=vocab, merges=tok["merges"], unk_token=vocab.unk_token or "<unk>")
    else:
        raise ValueError(f"Unknown tokenizer type in dataset: {tok['type']}")
    return data, tokenizer
