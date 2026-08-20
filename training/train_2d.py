"""2D ResNet50 training pipeline entrypoint.

Day 3 scope: build the pipeline and verify it runs (see
tests/test_train_2d.py, which exercises this against synthetic tensors,
not the real dataset). Running this file's __main__ against the real
Kaggle Chest X-ray Pneumonia dataset and reporting real metrics is a
Day 4 task, not Day 3 — do not fabricate results either way.
"""
import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.vision.dataset_2d import XrayDataset, build_patient_safe_splits, class_distribution
from src.vision.model_2d import build_resnet50, get_device


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = config["data"]
    train_cfg = config["training"]

    train_entries, val_entries, test_entries = build_patient_safe_splits(
        Path(data_cfg["dataset_root"]),
        val_fraction=data_cfg["val_fraction"],
        test_fraction=data_cfg["test_fraction"],
        seed=data_cfg["split_seed"],
    )

    print(f"Train class distribution: {class_distribution(train_entries)}")
    print(f"Val class distribution:   {class_distribution(val_entries)}")
    print(f"Test class distribution:  {class_distribution(test_entries)}")

    common = dict(batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"])
    train_loader = DataLoader(XrayDataset(train_entries), shuffle=True, **common)
    val_loader = DataLoader(XrayDataset(val_entries), shuffle=False, **common)
    test_loader = DataLoader(XrayDataset(test_entries), shuffle=False, **common)
    return train_loader, val_loader, test_loader


def build_model_and_optimizer(
    config: dict[str, Any], device: torch.device
) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    model_cfg = config["model"]
    train_cfg = config["training"]

    model = build_resnet50(
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        freeze_backbone=model_cfg["freeze_backbone"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    return model, optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in loader:
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_labels: list[int] = []
    all_probs: list[float] = []

    for images, labels in loader:
        images = images.to(device, dtype=torch.float32)
        labels_dev = labels.to(device, dtype=torch.long)

        logits = model(images)
        loss = loss_fn(logits, labels_dev)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]  # P(PNEUMONIA)
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

    metrics = {"loss": total_loss / max(n, 1)}

    # ROC-AUC needs both classes present; skip gracefully on tiny/degenerate
    # batches (e.g. synthetic smoke tests) rather than crashing the run.
    if len(set(all_labels)) > 1:
        from sklearn.metrics import roc_auc_score

        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def resolve_selection_metric(config: dict[str, Any]) -> str:
    """`configs/*.yaml`'s `evaluation.selection_metric` is written as
    `validation_<metric>` (e.g. `validation_roc_auc`); `evaluate()`'s
    returned metrics dict uses the bare key (e.g. `roc_auc`). Strips the
    `validation_` prefix to bridge the two, so `run_training` actually
    selects on what the config says instead of a hardcoded metric."""
    return config["evaluation"]["selection_metric"].removeprefix("validation_")


def is_better_metric(
    candidate: dict[str, float], best: dict[str, float], metric_key: str
) -> bool:
    """True if `candidate[metric_key]` improves on `best[metric_key]`.
    `best={}` means no checkpoint saved yet -- always an improvement.
    `roc_auc` is higher-is-better; everything else (currently just
    `loss`) is lower-is-better. A NaN candidate value (evaluate() returns
    NaN roc_auc for single-class batches, e.g. degenerate synthetic
    smoke tests) never counts as an improvement -- Python's `NaN > x` /
    `NaN < x` are both False, which already does the right thing, but
    this is spelled out explicitly rather than relied on implicitly."""
    if not best:
        return True
    value = candidate[metric_key]
    if value != value:  # NaN check without importing math
        return False
    if metric_key == "roc_auc":
        return value > best[metric_key]
    return value < best[metric_key]


def run_training(config: dict[str, Any]) -> dict[str, float]:
    """Runs the full training loop per `config` and returns the best
    validation metrics achieved. Used both by __main__ (real data) and
    by tests (synthetic data, via monkeypatched dataloaders)."""
    set_seed(config["training"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _test_loader = build_dataloaders(config)
    model, optimizer, scheduler = build_model_and_optimizer(config, device)
    loss_fn = nn.CrossEntropyLoss()

    checkpoint_path = Path(config["paths"]["checkpoint_dir"]) / config["paths"]["checkpoint_name"]
    selection_metric = resolve_selection_metric(config)
    epochs_without_improvement = 0
    best_metrics: dict[str, float] = {}

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_metrics["loss"])

        print(
            f"epoch {epoch + 1}/{config['training']['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_roc_auc={val_metrics['roc_auc']:.4f}"
        )

        if is_better_metric(val_metrics, best_metrics, selection_metric):
            best_metrics = val_metrics
            epochs_without_improvement = 0
            if config["evaluation"]["save_best_checkpoint"]:
                save_checkpoint(model, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["training"]["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 2D ResNet50 X-ray classifier")
    parser.add_argument("--config", default="configs/train_2d.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
