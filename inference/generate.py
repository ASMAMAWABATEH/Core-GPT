from __future__ import annotations

from typing import Optional

import torch

from models.gpt import GPT
from tokenizer.tokenizer import CharTokenizer


def generate_text(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    greedy: bool = False,
    device: str = "cpu",
) -> str:
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
    )
    return tokenizer.decode(output_ids[0].tolist())
