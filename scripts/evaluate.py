from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.preprocessing import load_processed_dataset
from datasets.text_dataset import TextDataset
from models.gpt import GPT, GPTConfig
from training.loss import language_modeling_loss
from utils.checkpoint import load_checkpoint
from utils.metrics import perplexity


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ScratchGPT")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--data_path", type=str, default="data/processed/tiny_shakespeare.pt")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    model_cfg = _load_yaml(PROJECT_ROOT / args.model_config)["model"]

    data, tokenizer = load_processed_dataset(PROJECT_ROOT / args.data_path)
    model_cfg["vocab_size"] = tokenizer.vocab_size

    split_idx = int(0.9 * data.size(0))
    val_data = data[split_idx:]
    val_ds = TextDataset(val_data, model_cfg["context_length"])
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT(GPTConfig(**model_cfg)).to(device)
    ckpt = load_checkpoint(Path(args.checkpoint), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss = language_modeling_loss(logits, y)
            total_loss += loss.item()
            total_batches += 1
            if args.max_batches is not None and total_batches >= args.max_batches:
                break

    avg_loss = total_loss / max(1, total_batches)
    print(f"val_loss={avg_loss:.4f} val_ppl={perplexity(avg_loss):.2f}")


if __name__ == "__main__":
    main()
