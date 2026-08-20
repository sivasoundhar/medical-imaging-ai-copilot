"""Day 7 test-set evaluation for the 3D LUNA16 nodule-candidate classifier.

Loads a trained checkpoint (training/checkpoints/model_3d_best.pth by
default) and runs it against the held-out TEST split -- never touched
during training/validation -- to report real, measured metrics. Per
CLAUDE.md: never fabricate metrics; only report numbers from an actual
run against real data.

Rebuilds the exact same series-safe test split training/train_3d.py used
(same config, same seed, see `build_test_entries_and_loader`) so there is
no leakage and the test set here matches what training carved out.
"""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from src.vision.dataset_3d import (
    CandidateEntry,
    CTCandidateDataset,
    available_series_ids,
    build_series_safe_splits,
    class_distribution,
    load_candidates,
)
from src.vision.model_2d import get_device
from src.vision.model_3d import build_model_3d
from training.train_2d import load_config


def build_test_entries_and_loader(
    config: dict[str, Any],
) -> tuple[list[CandidateEntry], DataLoader]:
    """Rebuilds the TEST split (entries, not just a loader) so predictions
    can be traced back to seriesuid for false-positives-per-scan --
    `CTCandidateDataset.__getitem__` only returns `(patch, label)`, not
    seriesuid, so entry order is tracked via `shuffle=False` loader order
    instead of changing that class's return contract."""
    data_cfg = config["data"]
    train_cfg = config["training"]

    dataset_root = Path(data_cfg["dataset_root"])
    available_series = available_series_ids(dataset_root)
    candidates = load_candidates(data_cfg["candidates_csv"], available_series=available_series)
    _, _, test_entries = build_series_safe_splits(
        candidates,
        val_fraction=data_cfg["val_fraction"],
        test_fraction=data_cfg["test_fraction"],
        seed=data_cfg["split_seed"],
    )
    print(f"Test class distribution: {class_distribution(test_entries)}")

    test_dataset = CTCandidateDataset(
        test_entries, dataset_root, crop_size=tuple(data_cfg["crop_size"])
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0
    )
    return test_entries, test_loader


@torch.no_grad()
def run_inference(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, probs) in loader iteration order -- matches
    test_entries order 1:1 since the loader is shuffle=False/num_workers=0."""
    model.eval()
    all_labels: list[int] = []
    all_probs: list[float] = []
    for images, labels in loader:
        images = images.to(device, dtype=torch.float32)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(nodule)
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())
    return np.array(all_labels), np.array(all_probs)


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    entries: list[CandidateEntry],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Candidate-level classification metrics plus the CT-specific
    false-positives-per-scan metric PROJECT_SPEC.md Day 7 asks for.
    `entries[i]` must correspond to `labels[i]`/`probs[i]` (same order)."""
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision == precision and sensitivity == sensitivity and (precision + sensitivity) > 0
        else float("nan")
    )

    has_both_classes = len(set(labels.tolist())) > 1
    metrics: dict[str, Any] = {
        "roc_auc": roc_auc_score(labels, probs) if has_both_classes else float("nan"),
        "pr_auc": average_precision_score(labels, probs) if has_both_classes else float("nan"),
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test_candidates": int(len(labels)),
        "decision_threshold": threshold,
    }

    # False positives per scan: for each distinct seriesuid in the test
    # split, count candidates predicted positive that are actually
    # non-nodules, then average across scans.
    per_series_fp: dict[str, int] = {}
    for entry, pred, label in zip(entries, preds.tolist(), labels.tolist()):
        per_series_fp.setdefault(entry.seriesuid, 0)
        if pred == 1 and label == 0:
            per_series_fp[entry.seriesuid] += 1
    n_scans = len(per_series_fp)
    metrics["false_positives_per_scan"] = (
        sum(per_series_fp.values()) / n_scans if n_scans > 0 else float("nan")
    )
    metrics["n_test_scans"] = n_scans

    return metrics


def write_report(metrics: dict[str, Any], checkpoint_path: Path, report_path: Path) -> None:
    lines = [
        "# 3D Nodule Classifier — Test Set Evaluation Report",
        "",
        f"Checkpoint: `{checkpoint_path}`",
        f"Test candidates: {metrics['n_test_candidates']} across {metrics['n_test_scans']} scans",
        "",
        "## Metrics",
        "",
        f"- ROC-AUC: {metrics['roc_auc']:.4f}",
        f"- PR-AUC: {metrics['pr_auc']:.4f}",
        f"- Sensitivity (recall): {metrics['sensitivity_recall']:.4f}",
        f"- Specificity: {metrics['specificity']:.4f}",
        f"- Precision: {metrics['precision']:.4f}",
        f"- F1: {metrics['f1']:.4f}",
        f"- False positives / scan: {metrics['false_positives_per_scan']:.4f}",
        f"- Decision threshold: {metrics['decision_threshold']}",
        "",
        "## Confusion matrix",
        "",
        f"- TP: {metrics['confusion_matrix']['tp']}",
        f"- FP: {metrics['confusion_matrix']['fp']}",
        f"- FN: {metrics['confusion_matrix']['fn']}",
        f"- TN: {metrics['confusion_matrix']['tn']}",
        "",
        "## Notes",
        "",
        "- Localization metrics (matching predictions to annotations.csv's "
        "real nodule coordinates) are NOT computed here -- candidates.csv "
        "already encodes candidate-vs-real-nodule labels, so sensitivity "
        "above is the nodule-detection-rate signal; per-nodule spatial "
        "localization scoring is deferred, not fabricated.",
        "- This is a research/portfolio prototype, not a clinical device. "
        "Metrics reflect LUNA16 subset0's held-out test split only.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the 3D CNN on the held-out test set")
    parser.add_argument("--config", default="configs/train_3d.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-json", default="evaluation/3d_metrics.json")
    parser.add_argument("--output-report", default="evaluation/3d_report.md")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = Path(
        args.checkpoint
        or Path(config["paths"]["checkpoint_dir"]) / config["paths"]["checkpoint_name"]
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path} -- run training/train_3d.py first."
        )

    device = get_device()
    print(f"Using device: {device}")

    test_entries, test_loader = build_test_entries_and_loader(config)

    model = build_model_3d(num_classes=config["model"]["num_classes"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    labels, probs = run_inference(model, test_loader, device)
    metrics = compute_metrics(labels, probs, test_entries)

    print(json.dumps(metrics, indent=2))

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    output_report = Path(args.output_report)
    write_report(metrics, checkpoint_path, output_report)
    print(f"Wrote {output_json} and {output_report}")


if __name__ == "__main__":
    main()
