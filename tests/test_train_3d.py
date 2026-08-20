"""Smoke tests for the 3D training pipeline — per Day 6's requirement,
these must run without the real LUNA16 dataset. Two layers:
  1. Pure-tensor smoke tests (mirrors test_train_2d.py), reusing
     train_2d.py's generic loop functions directly.
  2. A full build_dataloaders()/build_model_and_optimizer() smoke test
     against small synthetic .mhd volumes + a hand-written candidates.csv
     on disk — exercises the actual Day 6 entrypoint pieces, not just
     the reused generic loop.
"""
import numpy as np
import SimpleITK as sitk
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.vision.model_3d import build_model_3d
from training.train_2d import evaluate, save_checkpoint, train_one_epoch
from training.train_3d import (
    build_dataloaders,
    build_model_and_optimizer,
    build_weighted_sampler,
)
from training import train_3d as train_3d_module


class SyntheticCTDataset(Dataset):
    """Random tensors + random binary labels — no file I/O, no real data."""

    def __init__(self, n_samples: int = 8, crop_size=(16, 16, 16), seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.volumes = torch.randn(n_samples, 1, *crop_size, generator=generator)
        self.labels = torch.randint(0, 2, (n_samples,), generator=generator)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.volumes[index], self.labels[index].item()


def _write_synthetic_mhd(path, spacing=(1.0, 1.0, 1.0), size=(60, 60, 60), seed=0):
    rng = np.random.default_rng(seed=seed)
    array = rng.integers(-1000, 400, size=(size[2], size[1], size[0])).astype(np.int16)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path))


def _make_synthetic_dataset_dir(tmp_path, n_series: int = 4):
    """A small on-disk LUNA16-shaped fixture: a few synthetic volumes plus
    a matching candidates.csv, enough series to exercise a real
    train/val/test split (not just synthetic tensors in memory)."""
    dataset_root = tmp_path / "subset0"
    dataset_root.mkdir()
    candidates_csv = tmp_path / "candidates.csv"

    rows = ["seriesuid,coordX,coordY,coordZ,class"]
    for i in range(n_series):
        seriesuid = f"series-{i}"
        _write_synthetic_mhd(dataset_root / f"{seriesuid}.mhd", seed=i)
        # 2 non-nodule + 1 nodule candidate per series, all well inside
        # the volume so no edge-padding is involved here.
        rows.append(f"{seriesuid},10.0,10.0,10.0,0")
        rows.append(f"{seriesuid},20.0,20.0,20.0,0")
        rows.append(f"{seriesuid},30.0,30.0,30.0,1")
    candidates_csv.write_text("\n".join(rows) + "\n")

    return dataset_root, candidates_csv


def _make_config(dataset_root, candidates_csv):
    return {
        "data": {
            "dataset_root": str(dataset_root),
            "candidates_csv": str(candidates_csv),
            "val_fraction": 0.25,
            "test_fraction": 0.25,
            "split_seed": 0,
            "crop_size": [8, 8, 8],  # tiny — speed, not realism
        },
        "model": {"num_classes": 2},
        "training": {
            "batch_size": 2,
            "num_workers": 0,
            "balance_positive_fraction": 0.5,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
        },
    }


def test_train_one_epoch_runs_and_produces_finite_loss():
    device = torch.device("cpu")
    model = build_model_3d(num_classes=2).to(device)
    loader = DataLoader(SyntheticCTDataset(n_samples=8), batch_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    avg_loss = train_one_epoch(model, loader, optimizer, loss_fn, device)

    assert avg_loss == avg_loss  # not NaN
    assert avg_loss >= 0.0


def test_evaluate_runs_and_returns_expected_keys():
    device = torch.device("cpu")
    model = build_model_3d(num_classes=2).to(device)
    loader = DataLoader(SyntheticCTDataset(n_samples=8, seed=1), batch_size=4)
    loss_fn = nn.CrossEntropyLoss()

    metrics = evaluate(model, loader, loss_fn, device)

    assert "loss" in metrics
    assert "roc_auc" in metrics
    assert metrics["loss"] >= 0.0


def test_two_epoch_smoke_run_end_to_end():
    device = torch.device("cpu")
    model = build_model_3d(num_classes=2).to(device)
    train_loader = DataLoader(SyntheticCTDataset(n_samples=8, seed=2), batch_size=4)
    val_loader = DataLoader(SyntheticCTDataset(n_samples=8, seed=3), batch_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    loss_fn = nn.CrossEntropyLoss()

    for _epoch in range(2):
        train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_metrics["loss"])

    assert val_metrics["loss"] == val_metrics["loss"]  # not NaN


def test_checkpoint_save_creates_file(tmp_path):
    model = build_model_3d(num_classes=2)
    checkpoint_path = tmp_path / "checkpoints" / "model_3d_best.pth"

    save_checkpoint(model, checkpoint_path)

    assert checkpoint_path.exists()


def test_weighted_sampler_biases_toward_target_positive_fraction():
    entries_labels = [0] * 96 + [1] * 4  # ~4% positive, like real subset0 imbalance in spirit
    from src.vision.dataset_3d import CandidateEntry

    entries = [
        CandidateEntry(seriesuid=f"s{i}", coord_xyz=(0, 0, 0), label=label)
        for i, label in enumerate(entries_labels)
    ]

    sampler = build_weighted_sampler(entries, positive_fraction=0.5)
    drawn_labels = [entries[i].label for i in sampler]

    # Not an exact ratio (it's a random draw), but should be pulled far
    # above the ~4% base rate toward the 50% target.
    positive_rate = sum(drawn_labels) / len(drawn_labels)
    assert positive_rate > 0.25


def test_build_dataloaders_against_synthetic_on_disk_fixture(tmp_path):
    """Exercises the real Day 6 entrypoint pieces — CSV parsing, series-
    safe splitting, patch extraction from actual .mhd files on disk —
    without touching the real LUNA16 dataset."""
    dataset_root, candidates_csv = _make_synthetic_dataset_dir(tmp_path, n_series=4)
    config = _make_config(dataset_root, candidates_csv)

    train_loader, val_loader, test_loader = build_dataloaders(config)

    volumes, labels = next(iter(train_loader))
    assert volumes.shape[1:] == (1, 8, 8, 8)
    assert labels.shape[0] == volumes.shape[0]

    # every series-safe split combined accounts for all candidate rows
    total = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    assert total == 4 * 3  # 3 candidates per series, 4 series


def test_build_model_and_optimizer_produces_working_model(tmp_path):
    dataset_root, candidates_csv = _make_synthetic_dataset_dir(tmp_path, n_series=4)
    config = _make_config(dataset_root, candidates_csv)
    device = torch.device("cpu")

    model, optimizer, scheduler = build_model_and_optimizer(config, device)
    dummy_input = torch.randn(2, 1, 8, 8, 8)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 2)
    assert isinstance(optimizer, torch.optim.AdamW)


def test_run_training_selects_best_checkpoint_by_configured_metric_not_loss(monkeypatch):
    """Regression test for a real bug found from the actual LUNA16 Colab
    run (Day 7): `configs/train_3d.yaml` declares
    `selection_metric: validation_roc_auc`, but `run_training` used to
    hardcode selection on `val_loss` instead, ignoring the config
    entirely. Real observed epochs: epoch 5 had the run's best
    `val_roc_auc` (0.7597) but a mediocre `val_loss` (0.6276); epoch 7
    had a worse `val_roc_auc` (0.6618) but the lowest `val_loss`
    (0.0278) -- the old code saved epoch 7's checkpoint despite the
    config asking for the best-AUC one. Reproduces that exact shape here
    with mocked epochs, entirely on synthetic data (no GPU, no real
    dataset) -- `evaluate()` is monkeypatched to return a fixed metrics
    dict per call rather than actually computing anything."""
    epoch_metrics = [
        {"loss": 0.6276, "roc_auc": 0.7597},  # best roc_auc, mediocre loss
        {"loss": 0.0278, "roc_auc": 0.6618},  # best loss, worse roc_auc
        {"loss": 3.3892, "roc_auc": 0.6390},  # worse on both -- never selected
    ]
    call_count = {"evaluate": 0}

    def fake_evaluate(*_args, **_kwargs):
        metrics = epoch_metrics[call_count["evaluate"]]
        call_count["evaluate"] += 1
        return metrics

    class FakeScheduler:
        def step(self, _value):
            pass

    monkeypatch.setattr(train_3d_module, "build_dataloaders", lambda config: (None, None, None))
    monkeypatch.setattr(
        train_3d_module,
        "build_model_and_optimizer",
        lambda config, device: (build_model_3d(num_classes=2), None, FakeScheduler()),
    )
    monkeypatch.setattr(train_3d_module, "train_one_epoch", lambda *a, **k: 0.1)
    monkeypatch.setattr(train_3d_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(train_3d_module, "log_gpu_memory", lambda *a, **k: None)

    config = {
        "training": {"seed": 0, "epochs": 3, "early_stopping_patience": 5},
        "paths": {"checkpoint_dir": "unused", "checkpoint_name": "unused.pth"},
        "evaluation": {"selection_metric": "validation_roc_auc", "save_best_checkpoint": False},
    }

    best_metrics = train_3d_module.run_training(config)

    # Must be epoch 1's metrics (best roc_auc) -- NOT epoch 2's (best loss,
    # what the pre-fix code would have wrongly returned).
    assert best_metrics == epoch_metrics[0]
