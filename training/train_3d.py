"""3D CNN training pipeline entrypoint (LUNA16 nodule-candidate
classification).

Day 6 scope: build the pipeline and verify it runs against synthetic
volumes (see tests/test_train_3d.py) — running this file's __main__
against the real LUNA16 subset0 data and reporting real metrics is a
Day 7 task, not Day 6. Do not fabricate results either way.

Reuses training/train_2d.py's generic loop functions (`load_config`,
`set_seed`, `train_one_epoch`, `evaluate`, `save_checkpoint`) as-is —
none of them are X-ray-specific, they operate on generic
(images, labels) batches. Only the dataset/model-specific pieces
(dataloaders, model+optimizer, imbalance handling, GPU memory logging)
are new here. Does not modify train_2d.py.
"""
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.vision.dataset_3d import (
    CTCandidateDataset,
    available_series_ids,
    build_series_safe_splits,
    class_distribution,
    load_candidates,
)
from src.vision.model_2d import get_device
from src.vision.model_3d import build_model_3d, log_gpu_memory
from training.train_2d import (
    evaluate,
    is_better_metric,
    load_config,
    resolve_selection_metric,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


def build_weighted_sampler(entries: list, positive_fraction: float) -> WeightedRandomSampler:
    """Per-sample weights so batches drawn from `entries` average roughly
    `positive_fraction` nodule-positive — candidates.csv is ~466:1
    imbalanced on subset0 (see PROGRESS_LOG.md Day 6 entry), so uniform
    sampling would starve the model of positive examples almost entirely.
    Only meant for the train split; val/test stay unweighted so reported
    metrics reflect the real distribution, not an oversampled one."""
    labels = np.array([e.label for e in entries])
    n_pos = max(1, int((labels == 1).sum()))
    n_neg = max(1, int((labels == 0).sum()))
    pos_weight = positive_fraction / n_pos
    neg_weight = (1 - positive_fraction) / n_neg
    weights = np.where(labels == 1, pos_weight, neg_weight)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(entries),
        replacement=True,
    )


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = config["data"]
    train_cfg = config["training"]

    dataset_root = Path(data_cfg["dataset_root"])
    available_series = available_series_ids(dataset_root)
    if not available_series:
        raise FileNotFoundError(f"No .mhd volumes found under {dataset_root}")

    candidates = load_candidates(data_cfg["candidates_csv"], available_series=available_series)
    train_entries, val_entries, test_entries = build_series_safe_splits(
        candidates,
        val_fraction=data_cfg["val_fraction"],
        test_fraction=data_cfg["test_fraction"],
        seed=data_cfg["split_seed"],
    )

    print(f"Train class distribution: {class_distribution(train_entries)}")
    print(f"Val class distribution:   {class_distribution(val_entries)}")
    print(f"Test class distribution:  {class_distribution(test_entries)}")

    crop_size = tuple(data_cfg["crop_size"])
    num_workers = train_cfg["num_workers"]
    common = dict(
        batch_size=train_cfg["batch_size"],
        num_workers=num_workers,
        # DataLoader tears down and recreates worker processes every
        # epoch by default, which would throw away CTCandidateDataset's
        # per-series LRU cache (see its docstring) at the start of every
        # single epoch. persistent_workers keeps them alive across
        # epochs instead. Only valid when num_workers > 0.
        persistent_workers=num_workers > 0,
    )

    train_dataset = CTCandidateDataset(train_entries, dataset_root, crop_size=crop_size)
    val_dataset = CTCandidateDataset(val_entries, dataset_root, crop_size=crop_size)
    test_dataset = CTCandidateDataset(test_entries, dataset_root, crop_size=crop_size)

    sampler = build_weighted_sampler(train_entries, train_cfg["balance_positive_fraction"])
    train_loader = DataLoader(train_dataset, sampler=sampler, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def build_model_and_optimizer(
    config: dict[str, Any], device: torch.device
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    model_cfg = config["model"]
    train_cfg = config["training"]

    model = build_model_3d(num_classes=model_cfg["num_classes"]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    return model, optimizer, scheduler


def run_training(config: dict[str, Any]) -> dict[str, float]:
    """Runs the full training loop per `config` and returns the best
    validation metrics achieved. Used both by __main__ (real data) and
    by tests (synthetic data, via monkeypatched dataloaders)."""
    set_seed(config["training"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _test_loader = build_dataloaders(config)
    model, optimizer, scheduler = build_model_and_optimizer(config, device)
    loss_fn = torch.nn.CrossEntropyLoss()
    log_gpu_memory(device, prefix="After model init: ")

    checkpoint_path = Path(config["paths"]["checkpoint_dir"]) / config["paths"]["checkpoint_name"]
    selection_metric = resolve_selection_metric(config)
    epochs_without_improvement = 0
    best_metrics: dict[str, float] = {}

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        log_gpu_memory(device, prefix=f"After epoch {epoch + 1} train: ")
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
    parser = argparse.ArgumentParser(description="Train the 3D CNN LUNA16 nodule classifier")
    parser.add_argument("--config", default="configs/train_3d.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    best_metrics = run_training(config)
    print(f"Best val metrics ({resolve_selection_metric(config)}): {best_metrics}")


if __name__ == "__main__":
    main()
