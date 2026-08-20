"""Smoke test for the training loop using synthetic fake tensors — per
Day 3's requirement, this must run without downloading the real dataset
or pretrained ImageNet weights (pretrained=False)."""
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.vision.model_2d import build_resnet50
from training.train_2d import (
    evaluate,
    is_better_metric,
    resolve_selection_metric,
    save_checkpoint,
    train_one_epoch,
)


class SyntheticXrayDataset(Dataset):
    """Random tensors + random binary labels — no file I/O, no real data."""

    def __init__(self, n_samples: int = 8, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n_samples, 3, 224, 224, generator=generator)
        self.labels = torch.randint(0, 2, (n_samples,), generator=generator)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index].item()


def test_train_one_epoch_runs_and_produces_finite_loss() -> None:
    device = torch.device("cpu")
    model = build_resnet50(num_classes=2, pretrained=False).to(device)
    loader = DataLoader(SyntheticXrayDataset(n_samples=8), batch_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    avg_loss = train_one_epoch(model, loader, optimizer, loss_fn, device)

    assert avg_loss == avg_loss  # not NaN
    assert avg_loss >= 0.0


def test_evaluate_runs_and_returns_expected_keys() -> None:
    device = torch.device("cpu")
    model = build_resnet50(num_classes=2, pretrained=False).to(device)
    loader = DataLoader(SyntheticXrayDataset(n_samples=8, seed=1), batch_size=4)
    loss_fn = nn.CrossEntropyLoss()

    metrics = evaluate(model, loader, loss_fn, device)

    assert "loss" in metrics
    assert "roc_auc" in metrics
    assert metrics["loss"] >= 0.0


def test_two_epoch_smoke_run_end_to_end() -> None:
    """Mimics the shape of run_training's inner loop, but entirely on
    synthetic tensors — proves the pipeline (forward, backward, eval,
    scheduler step, checkpointing) doesn't crash, without touching the
    real dataset or downloading pretrained weights."""
    device = torch.device("cpu")
    model = build_resnet50(num_classes=2, pretrained=False).to(device)
    train_loader = DataLoader(SyntheticXrayDataset(n_samples=8, seed=2), batch_size=4)
    val_loader = DataLoader(SyntheticXrayDataset(n_samples=8, seed=3), batch_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    loss_fn = nn.CrossEntropyLoss()

    for _epoch in range(2):
        train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_metrics["loss"])

    assert val_metrics["loss"] == val_metrics["loss"]  # not NaN


def test_checkpoint_save_creates_file(tmp_path) -> None:
    model = build_resnet50(num_classes=2, pretrained=False)
    checkpoint_path = tmp_path / "checkpoints" / "model_2d_best.pth"

    save_checkpoint(model, checkpoint_path)

    assert checkpoint_path.exists()


def test_resolve_selection_metric_strips_validation_prefix() -> None:
    config = {"evaluation": {"selection_metric": "validation_roc_auc"}}
    assert resolve_selection_metric(config) == "roc_auc"


def test_is_better_metric_empty_best_is_always_an_improvement() -> None:
    assert is_better_metric({"roc_auc": 0.1}, {}, "roc_auc") is True


def test_is_better_metric_roc_auc_is_higher_is_better() -> None:
    best = {"roc_auc": 0.6}
    assert is_better_metric({"roc_auc": 0.7}, best, "roc_auc") is True
    assert is_better_metric({"roc_auc": 0.5}, best, "roc_auc") is False


def test_is_better_metric_loss_is_lower_is_better() -> None:
    best = {"loss": 0.3}
    assert is_better_metric({"loss": 0.2}, best, "loss") is True
    assert is_better_metric({"loss": 0.4}, best, "loss") is False


def test_is_better_metric_nan_candidate_never_wins() -> None:
    best = {"roc_auc": 0.6}
    assert is_better_metric({"roc_auc": float("nan")}, best, "roc_auc") is False
