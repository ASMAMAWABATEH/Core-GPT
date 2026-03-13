from __future__ import annotations

import math


def perplexity(loss: float) -> float:
    return float(math.exp(loss)) if loss < 20 else float("inf")
