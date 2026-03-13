from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .embedding import TokenEmbedding
from .positional_encoding import PositionalEmbedding
from .transformer_block import TransformerBlock
from inference.sampling import sample_next_token


@dataclass
class GPTConfig:
    vocab_size: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    context_length: int
    dropout: float = 0.1
    use_bias: bool = True


class GPT(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = TokenEmbedding(config.vocab_size, config.embedding_dim)
        self.pos_embedding = PositionalEmbedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=config.embedding_dim,
                    num_heads=config.num_heads,
                    context_length=config.context_length,
                    dropout=config.dropout,
                    use_bias=config.use_bias,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.context_length:
            raise ValueError("Sequence length exceeds context length")

        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(input_ids)
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        greedy: bool = False,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.context_length :]
            logits, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :]
            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
            )
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
