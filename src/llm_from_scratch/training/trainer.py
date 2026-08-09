from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from llm_from_scratch.data.dataset import CharDataset, get_batch
from llm_from_scratch.nn.gpt import GPT
from llm_from_scratch.training.config import TrainConfig


def compute_loss(model: GPT, x: Tensor, y: Tensor) -> Tensor:
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


@torch.no_grad()
def evaluate(model: GPT, dataset: CharDataset, config: TrainConfig, device: torch.device) -> float:
    model.eval()
    losses = [compute_loss(model, *get_batch(dataset, config.batch_size, device)).item() for _ in range(config.eval_steps)]
    model.train()
    return sum(losses) / len(losses)


def train_model(
    model: GPT,
    train_dataset: CharDataset,
    val_dataset: CharDataset,
    config: TrainConfig,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> GPT:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for step in range(config.max_steps):
        x, y = get_batch(train_dataset, config.batch_size, device)
        loss = compute_loss(model, x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.eval_interval == 0 or step == config.max_steps - 1:
            val_loss = evaluate(model, val_dataset, config, device)
            print(f"step {step}: train loss {loss.item():.4f}, val loss {val_loss:.4f}")

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": model.config}, checkpoint_path)

    return model
