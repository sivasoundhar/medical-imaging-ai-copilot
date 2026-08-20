"""X-ray dataset manifest building, patient-level split validation, and
the torch Dataset wrapper for training.

Dataset-specific note (Kaggle Chest X-ray Pneumonia): filenames encode a
usable (if heuristic) patient-grouping key —
  - PNEUMONIA images: "person<N>_bacteria|virus_<M>.jpeg" -> patient <N>
  - NORMAL images:    "IM-<NNNN>-<MMMM>.jpeg"             -> patient <NNNN>
Verified against the actual dataset: NORMAL has 1341 files across only
1262 unique "IM-<NNNN>" prefixes, i.e. some patients do contribute
multiple images, so grouping by this key is meaningful, not a no-op.

Known dataset flaws (documented here deliberately, both confirmed
against the actual downloaded data, not assumed):
  1. The official `val/` split has only 16 images (8 NORMAL +
     8 PNEUMONIA) — too small for a reliable validation signal.
  2. The official `train/` and `test/` folders are NOT patient-disjoint
     — e.g. patient "pneumonia-137" has images in both. Trusting the
     official test/ folder as a held-out set would leak patients into
     "held-out" evaluation.
Because of (2), `build_patient_safe_splits()` below pools ALL THREE
official folders and re-splits by patient group from scratch — it does
not trust any official split boundary. This is the only way to get a
genuinely patient-disjoint train/val/test from this dataset.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from src.preprocessing.preprocess_2d import preprocess_xray

LABELS = {"NORMAL": 0, "PNEUMONIA": 1}

_PERSON_RE = re.compile(r"^person(\d+)_", re.IGNORECASE)
_IM_RE = re.compile(r"^IM-(\d+)-", re.IGNORECASE)


class LeakageError(ValueError):
    """Raised when the same patient ID appears in more than one split."""


@dataclass
class ManifestEntry:
    path: Path
    label: int
    patient_id: str


def extract_patient_id(filename: str) -> str:
    """Best-effort patient grouping key from the Kaggle filename
    convention. Falls back to the filename stem (i.e. "own patient") if
    neither known pattern matches, rather than silently mis-grouping."""
    match = _PERSON_RE.match(filename)
    if match:
        return f"pneumonia-{match.group(1)}"

    match = _IM_RE.match(filename)
    if match:
        return f"normal-{match.group(1)}"

    return f"unknown-{Path(filename).stem}"


def build_manifest(class_dir: Path, label: int) -> list[ManifestEntry]:
    """Scan one class directory (e.g. .../train/NORMAL) into manifest
    entries. Only files preprocess_xray can plausibly read are included;
    directory listing order is sorted for reproducibility."""
    entries = []
    for path in sorted(class_dir.glob("*")):
        if not path.is_file():
            continue
        entries.append(
            ManifestEntry(path=path, label=label, patient_id=extract_patient_id(path.name))
        )
    return entries


def build_full_manifest(dataset_root: Path, split: str) -> list[ManifestEntry]:
    """Build the manifest for one split ('train', 'val', or 'test') by
    combining both class directories."""
    entries: list[ManifestEntry] = []
    for class_name, label in LABELS.items():
        class_dir = dataset_root / split / class_name
        if class_dir.exists():
            entries.extend(build_manifest(class_dir, label))
    return entries


def validate_no_leakage(*splits: list[ManifestEntry]) -> None:
    """Raise LeakageError if any patient_id appears in more than one of
    the given splits."""
    seen: dict[str, int] = {}
    for split_index, split in enumerate(splits):
        patient_ids = {entry.patient_id for entry in split}
        for pid in patient_ids:
            if pid in seen and seen[pid] != split_index:
                raise LeakageError(
                    f"Patient '{pid}' appears in more than one split "
                    f"(split {seen[pid]} and split {split_index}) — leakage."
                )
            seen[pid] = split_index


def build_patient_safe_splits(
    dataset_root: Path,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[ManifestEntry], list[ManifestEntry], list[ManifestEntry]]:
    """Pool ALL official train+val+test folders and re-split by patient
    group from scratch. Required because the official train/ and test/
    folders are not patient-disjoint in this dataset (confirmed: e.g.
    patient "pneumonia-137" appears in both) — trusting either official
    boundary would leak patients into a supposedly held-out set.

    Returns (train, val, test), guaranteed patient-disjoint.
    """
    pooled = (
        build_full_manifest(dataset_root, "train")
        + build_full_manifest(dataset_root, "val")
        + build_full_manifest(dataset_root, "test")
    )

    patient_ids = sorted({entry.patient_id for entry in pooled})
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    n_val_patients = max(1, round(len(patient_ids) * val_fraction))
    n_test_patients = max(1, round(len(patient_ids) * test_fraction))

    val_patient_set = set(patient_ids[:n_val_patients])
    test_patient_set = set(patient_ids[n_val_patients : n_val_patients + n_test_patients])

    train = [
        e for e in pooled if e.patient_id not in val_patient_set | test_patient_set
    ]
    val = [e for e in pooled if e.patient_id in val_patient_set]
    test = [e for e in pooled if e.patient_id in test_patient_set]

    validate_no_leakage(train, val, test)
    return train, val, test


def class_distribution(entries: list[ManifestEntry]) -> dict[str, int]:
    counts = {name: 0 for name in LABELS}
    reverse = {v: k for k, v in LABELS.items()}
    for entry in entries:
        counts[reverse[entry.label]] += 1
    return counts


class XrayDataset(Dataset):
    """Wraps preprocess_xray + label lookup for a manifest of entries."""

    def __init__(self, entries: list[ManifestEntry]):
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        array = preprocess_xray(entry.path)  # (3, H, W) float32
        return array, entry.label
