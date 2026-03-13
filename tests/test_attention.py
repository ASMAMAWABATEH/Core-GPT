import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from scratchgpt.models.attention import MultiHeadSelfAttention


def test_causal_mask_is_lower_triangular():
    attn = MultiHeadSelfAttention(
        embedding_dim=8,
        num_heads=2,
        context_length=4,
        dropout=0.0,
    )
    mask = attn.causal_mask[0, 0]
    expected = torch.tril(torch.ones(4, 4))
    assert torch.equal(mask, expected)
