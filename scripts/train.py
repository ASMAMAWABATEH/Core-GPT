from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from scratchgpt.datasets.preprocessing import load_processed_dataset
from scratchgpt.datasets.text_dataset import TextDataset
from scratchgpt.models.gpt import GPT, GPTConfig
from scratchgpt.training.optimizer import build_optimizer
from scratchgpt.training.scheduler import build_scheduler
from scratchgpt.training.trainer import Trainer, TrainingConfig
from scratchgpt.utils.checkpoint import load_checkpoint
from scratchgpt.utils.logger import Logger
from scratchgpt.utils.seed import set_seed


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ScratchGPT")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--training_config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--data_path", type=str, default="data/processed/tiny_shakespeare.pt")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    model_cfg = _load_yaml(PROJECT_ROOT / args.model_config)["model"]
    train_cfg = _load_yaml(PROJECT_ROOT / args.training_config)["training"]

    data, vocab = load_processed_dataset(PROJECT_ROOT / args.data_path)
    model_cfg["vocab_size"] = vocab.size

    device = train_cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg["device"] = device

    set_seed(train_cfg["seed"])

    split_idx = int(0.9 * data.size(0))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_ds = TextDataset(train_data, model_cfg["context_length"])
    val_ds = TextDataset(val_data, model_cfg["context_length"])

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        drop_last=False,
    )

    config = GPTConfig(**model_cfg)
    model = GPT(config)
    start_step = 0

    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        start_step = int(ckpt.get("step", 0)) + 1

    model.to(device)

    optimizer = build_optimizer(model.parameters(), train_cfg["learning_rate"], train_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, train_cfg["warmup_steps"], train_cfg["max_training_steps"])

    if args.resume:
        if ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
        if ckpt.get("scheduler_state") is not None and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=TrainingConfig(**train_cfg),
        logger=Logger(),
        start_step=start_step,
    )
    trainer.train()


if __name__ == "__main__":
    main()
