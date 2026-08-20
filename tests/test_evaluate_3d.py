"""Tests for Day 7 test-set evaluation (evaluation/evaluate_3d.py).

Only `compute_metrics` is unit-tested here (pure function over
labels/probs/entries) -- `build_test_entries_and_loader` and
`run_inference` are thin wrappers over already-tested pieces
(`build_series_safe_splits`, `CTCandidateDataset`, a plain DataLoader
loop) and don't need duplicate coverage. Synthetic data only, per
project convention.
"""
import numpy as np

from evaluation.evaluate_3d import compute_metrics
from src.vision.dataset_3d import CandidateEntry


def _entry(seriesuid: str, label: int) -> CandidateEntry:
    return CandidateEntry(seriesuid=seriesuid, coord_xyz=(0.0, 0.0, 0.0), label=label)


def test_compute_metrics_perfect_separation_gives_auc_one_and_no_errors():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    entries = [_entry("s1", 0), _entry("s1", 0), _entry("s2", 1), _entry("s2", 1)]

    metrics = compute_metrics(labels, probs, entries)

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["sensitivity_recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert metrics["n_test_candidates"] == 4


def test_compute_metrics_confusion_matrix_matches_known_errors():
    # 1 false positive, 1 false negative, rest correct.
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.6, 0.1, 0.4, 0.9])  # idx0 FP, idx2 FN
    entries = [_entry("s1", label) for label in labels]

    metrics = compute_metrics(labels, probs, entries)

    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["sensitivity_recall"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["precision"] == 0.5


def test_compute_metrics_false_positives_per_scan_averages_across_distinct_series():
    # series A: 2 candidates, 1 real FP. series B: 2 candidates, 0 FP.
    labels = np.array([0, 0, 0, 1])
    probs = np.array([0.9, 0.1, 0.1, 0.9])  # idx0 (series A) is a FP
    entries = [_entry("A", 0), _entry("A", 0), _entry("B", 0), _entry("B", 1)]

    metrics = compute_metrics(labels, probs, entries)

    assert metrics["n_test_scans"] == 2
    # 1 total FP across 2 scans -> 0.5/scan.
    assert metrics["false_positives_per_scan"] == 0.5


def test_compute_metrics_single_class_labels_returns_nan_auc_not_a_crash():
    labels = np.array([0, 0, 0])
    probs = np.array([0.1, 0.2, 0.3])
    entries = [_entry("s1", 0), _entry("s1", 0), _entry("s2", 0)]

    metrics = compute_metrics(labels, probs, entries)

    assert metrics["roc_auc"] != metrics["roc_auc"]  # NaN
    assert metrics["pr_auc"] != metrics["pr_auc"]  # NaN
    assert metrics["confusion_matrix"] == {"tn": 3, "fp": 0, "fn": 0, "tp": 0}


def test_compute_metrics_respects_custom_threshold():
    labels = np.array([0, 1])
    probs = np.array([0.4, 0.4])
    entries = [_entry("s1", 0), _entry("s1", 1)]

    # threshold 0.5: both predicted negative -> 1 FN, 0 FP.
    default_metrics = compute_metrics(labels, probs, entries, threshold=0.5)
    assert default_metrics["confusion_matrix"] == {"tn": 1, "fp": 0, "fn": 1, "tp": 0}

    # threshold 0.3: both predicted positive -> 1 TP, 1 FP.
    low_threshold_metrics = compute_metrics(labels, probs, entries, threshold=0.3)
    assert low_threshold_metrics["confusion_matrix"] == {"tn": 0, "fp": 1, "fn": 0, "tp": 1}
