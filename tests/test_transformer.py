import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from scratchgpt.models.gpt import GPT, GPTConfig


def test_gpt_forward_shapes():
    config = GPTConfig(
        vocab_size=20,
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        context_length=8,
        dropout=0.0,
    )
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_length))
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, config.context_length, config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
