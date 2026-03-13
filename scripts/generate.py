from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from scratchgpt.datasets.preprocessing import load_processed_dataset
from scratchgpt.inference.generate import generate_text
from scratchgpt.models.gpt import GPT, GPTConfig
from scratchgpt.tokenizer.tokenizer import CharTokenizer
from scratchgpt.utils.checkpoint import load_checkpoint


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with ScratchGPT")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--data_path", type=str, default="data/processed/tiny_shakespeare.pt")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    model_cfg = _load_yaml(PROJECT_ROOT / args.model_config)["model"]

    data, vocab = load_processed_dataset(PROJECT_ROOT / args.data_path)
    model_cfg["vocab_size"] = vocab.size

    tokenizer = CharTokenizer(vocab=vocab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(GPTConfig(**model_cfg)).to(device)
    ckpt = load_checkpoint(Path(args.checkpoint), map_location=device)
    model.load_state_dict(ckpt["model_state"])

    text = generate_text(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
