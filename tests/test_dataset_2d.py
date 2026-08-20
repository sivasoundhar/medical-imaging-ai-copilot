"""Tests for manifest building and patient-level split validation.
Uses a small synthetic directory tree (fake tiny images), not the real
dataset."""
import numpy as np
import pytest
from PIL import Image

from src.vision.dataset_2d import (
    LeakageError,
    ManifestEntry,
    build_full_manifest,
    build_patient_safe_splits,
    class_distribution,
    extract_patient_id,
    validate_no_leakage,
)


def _write_fake_jpeg(path) -> None:
    array = np.zeros((20, 20), dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path)


def _make_synthetic_dataset_root(tmp_path):
    """Mimics Data/chest_xray/{split}/{class}/*.jpeg, with a few patients
    contributing multiple images so patient-grouping is actually
    exercised, not trivially 1 image = 1 patient."""
    root = tmp_path / "chest_xray"
    layout = {
        "train": {
            "NORMAL": ["IM-0001-0001.jpeg", "IM-0001-0002.jpeg", "IM-0002-0001.jpeg"],
            "PNEUMONIA": ["person1_bacteria_1.jpeg", "person1_bacteria_2.jpeg", "person2_virus_1.jpeg"],
        },
        "val": {
            "NORMAL": ["IM-0003-0001.jpeg"],
            "PNEUMONIA": ["person3_bacteria_1.jpeg"],
        },
        "test": {
            "NORMAL": ["IM-0099-0001.jpeg"],
            "PNEUMONIA": ["person99_bacteria_1.jpeg"],
        },
    }
    for split, classes in layout.items():
        for class_name, filenames in classes.items():
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            for name in filenames:
                _write_fake_jpeg(class_dir / name)
    return root


def test_extract_patient_id_pneumonia_pattern() -> None:
    assert extract_patient_id("person1000_bacteria_2931.jpeg") == "pneumonia-1000"


def test_extract_patient_id_normal_pattern() -> None:
    assert extract_patient_id("IM-0115-0001.jpeg") == "normal-0115"


def test_extract_patient_id_fallback_for_unknown_pattern() -> None:
    assert extract_patient_id("weird_file_name.jpeg") == "unknown-weird_file_name"


def test_validate_no_leakage_passes_for_disjoint_patients() -> None:
    train = [ManifestEntry(path="a", label=0, patient_id="p1")]
    val = [ManifestEntry(path="b", label=0, patient_id="p2")]
    validate_no_leakage(train, val)  # should not raise


def test_validate_no_leakage_raises_for_shared_patient() -> None:
    train = [ManifestEntry(path="a", label=0, patient_id="p1")]
    val = [ManifestEntry(path="b", label=1, patient_id="p1")]  # same patient, different split
    with pytest.raises(LeakageError):
        validate_no_leakage(train, val)


def test_build_full_manifest_counts(tmp_path) -> None:
    root = _make_synthetic_dataset_root(tmp_path)
    train_entries = build_full_manifest(root, "train")
    assert len(train_entries) == 6  # 3 NORMAL + 3 PNEUMONIA

    dist = class_distribution(train_entries)
    assert dist == {"NORMAL": 3, "PNEUMONIA": 3}


def test_build_patient_safe_splits_is_patient_disjoint(tmp_path) -> None:
    root = _make_synthetic_dataset_root(tmp_path)
    train, val, test = build_patient_safe_splits(root, val_fraction=0.2, test_fraction=0.2, seed=0)

    # Repeats validate_no_leakage internally already, but assert directly too.
    train_patients = {e.patient_id for e in train}
    val_patients = {e.patient_id for e in val}
    test_patients = {e.patient_id for e in test}

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)

    # every image from the pooled train+val+test ends up in exactly one split
    assert len(train) + len(val) + len(test) == 10  # all 10 fixture images accounted for


def test_build_patient_safe_splits_detects_cross_split_leakage_in_official_folders(
    tmp_path,
) -> None:
    """Regression test for the real bug this caught: if official train/
    and test/ folders shared a patient, a naive split trusting official
    boundaries would leak. Pooling-and-resplitting must not reproduce
    that leak."""
    root = _make_synthetic_dataset_root(tmp_path)
    # Duplicate a train patient's image into test/, mimicking the real
    # dataset's confirmed train/test patient overlap.
    leaking_file = root / "test" / "NORMAL" / "IM-0001-0099.jpeg"
    _write_fake_jpeg(leaking_file)

    train, val, test = build_patient_safe_splits(root, val_fraction=0.2, test_fraction=0.2, seed=0)

    train_patients = {e.patient_id for e in train}
    val_patients = {e.patient_id for e in val}
    test_patients = {e.patient_id for e in test}
    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)
