from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.loss import language_modeling_loss
from utils.checkpoint import save_checkpoint
from utils.logger import Logger
from utils.metrics import perplexity


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    max_training_steps: int
    eval_interval: int
    save_interval: int
    grad_clip: float
    seed: int
    device: str
    weight_decay: float
    num_workers: int
    log_interval: int
    warmup_steps: int
    checkpoint_dir: str


class Trainer:
    """Simple training loop for autoregressive language modeling."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainingConfig,
        logger: Optional[Logger] = None,
        start_step: int = 0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger or Logger()
        self.device = torch.device(config.device)
        self.start_step = start_step

    def _evaluate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {"val_loss": float("nan"), "val_ppl": float("nan")}
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits, _ = self.model(x)
                loss = language_modeling_loss(logits, y)
                total_loss += loss.item()
                total_batches += 1
        avg_loss = total_loss / max(1, total_batches)
        return {"val_loss": avg_loss, "val_ppl": perplexity(avg_loss)}

    def train(self) -> None:
        self.model.to(self.device)
        self.model.train()
        step = self.start_step
        loader_iter = iter(self.train_loader)

        pbar = tqdm(total=self.config.max_training_steps, desc="training")
        while step < self.config.max_training_steps:
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                x, y = next(loader_iter)

            x = x.to(self.device)
            y = y.to(self.device)

            logits, _ = self.model(x)
            loss = language_modeling_loss(logits, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            if step % self.config.log_interval == 0:
                self.logger.info(
                    f"step={step} loss={loss.item():.4f} ppl={perplexity(loss.item()):.2f}"
                )

            if step % self.config.eval_interval == 0 and step > 0:
                metrics = self._evaluate()
                self.logger.info(
                    f"eval step={step} val_loss={metrics['val_loss']:.4f} val_ppl={metrics['val_ppl']:.2f}"
                )
                self.model.train()

            if step % self.config.save_interval == 0 and step > 0:
                ckpt_path = Path(self.config.checkpoint_dir) / f"step_{step}.pt"
                save_checkpoint(
                    ckpt_path,
                    model_state=self.model.state_dict(),
                    optimizer_state=self.optimizer.state_dict(),
                    scheduler_state=self.scheduler.state_dict() if self.scheduler is not None else None,
                    step=step,
                    config={"training": self.config.__dict__},
                )
                self.logger.info(f"saved checkpoint to {ckpt_path}")

            step += 1
            pbar.update(1)
        pbar.close()

        final_step = self.config.max_training_steps
        final_ckpt = Path(self.config.checkpoint_dir) / f"step_{final_step}.pt"
        save_checkpoint(
            final_ckpt,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict() if self.scheduler is not None else None,
            step=final_step,
            config={"training": self.config.__dict__},
        )
        self.logger.info(f"saved final checkpoint to {final_ckpt}")
