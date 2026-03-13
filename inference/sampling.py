from __future__ import annotations

from typing import Optional

import torch


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    vocab_size = logits.size(-1)
    top_k = min(top_k, vocab_size)
    values, _ = torch.topk(logits, top_k)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > top_p
    sorted_mask[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
    original_logits = torch.full_like(logits, float("-inf"))
    original_logits.scatter_(1, sorted_indices, sorted_logits)
    return original_logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    greedy: bool = False,
) -> torch.Tensor:
    if greedy:
        return greedy_next_token(logits)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits / temperature
    if top_k is not None:
        logits = top_k_filtering(logits, top_k)
    if top_p is not None:
        logits = top_p_filtering(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def greedy_next_token(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1, keepdim=True)
