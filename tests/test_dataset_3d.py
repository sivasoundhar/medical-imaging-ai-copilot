"""Tests for LUNA16 candidate manifest building, series-level split
validation, and 3D patch extraction. Synthetic in-memory volumes and a
hand-written candidates.csv fixture only — never real patient data."""
import numpy as np
import pytest
import SimpleITK as sitk

import src.vision.dataset_3d as dataset_3d
from src.vision.dataset_3d import (
    CandidateEntry,
    CTCandidateDataset,
    LeakageError,
    available_series_ids,
    build_series_safe_splits,
    class_distribution,
    extract_candidate_patch,
    load_candidates,
    validate_no_leakage,
)


def _write_synthetic_mhd(tmp_path, seriesuid: str, size=(60, 60, 60), spacing=(1.0, 1.0, 1.0)):
    """Isotropic 1mm synthetic volume with origin (0,0,0) — keeps
    world-coordinate math trivial (world mm == voxel index) for
    predictable test assertions."""
    rng = np.random.default_rng(seed=0)
    array = rng.integers(-1000, 400, size=(size[2], size[1], size[0])).astype(np.int16)  # (D,H,W)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    path = tmp_path / f"{seriesuid}.mhd"
    sitk.WriteImage(image, str(path))
    return path


def _write_candidates_csv(path, rows: list[tuple[str, float, float, float, int]]) -> None:
    with open(path, "w") as f:
        f.write("seriesuid,coordX,coordY,coordZ,class\n")
        for seriesuid, x, y, z, label in rows:
            f.write(f"{seriesuid},{x},{y},{z},{label}\n")


def test_available_series_ids_reads_mhd_stems(tmp_path):
    _write_synthetic_mhd(tmp_path, "series-a")
    _write_synthetic_mhd(tmp_path, "series-b")

    assert available_series_ids(tmp_path) == {"series-a", "series-b"}


def test_load_candidates_filters_to_available_series(tmp_path):
    csv_path = tmp_path / "candidates.csv"
    _write_candidates_csv(
        csv_path,
        [
            ("series-a", 1.0, 2.0, 3.0, 0),
            ("series-b", 4.0, 5.0, 6.0, 1),
            ("series-not-local", 7.0, 8.0, 9.0, 0),
        ],
    )

    entries = load_candidates(csv_path, available_series={"series-a", "series-b"})

    assert {e.seriesuid for e in entries} == {"series-a", "series-b"}
    assert len(entries) == 2


def test_load_candidates_without_filter_keeps_all_rows(tmp_path):
    csv_path = tmp_path / "candidates.csv"
    _write_candidates_csv(csv_path, [("series-a", 1.0, 2.0, 3.0, 0)])

    entries = load_candidates(csv_path)

    assert len(entries) == 1
    assert entries[0].coord_xyz == (1.0, 2.0, 3.0)
    assert entries[0].label == 0


def test_validate_no_leakage_passes_for_disjoint_series():
    train = [CandidateEntry(seriesuid="s1", coord_xyz=(0, 0, 0), label=0)]
    val = [CandidateEntry(seriesuid="s2", coord_xyz=(0, 0, 0), label=0)]
    validate_no_leakage(train, val)  # should not raise


def test_validate_no_leakage_raises_for_shared_series():
    train = [CandidateEntry(seriesuid="s1", coord_xyz=(0, 0, 0), label=0)]
    val = [CandidateEntry(seriesuid="s1", coord_xyz=(1, 1, 1), label=1)]  # same series
    with pytest.raises(LeakageError):
        validate_no_leakage(train, val)


def test_build_series_safe_splits_is_series_disjoint():
    candidates = [
        CandidateEntry(seriesuid=f"s{i}", coord_xyz=(0, 0, 0), label=i % 2) for i in range(10)
    ]
    train, val, test = build_series_safe_splits(
        candidates, val_fraction=0.2, test_fraction=0.2, seed=0
    )

    train_series = {e.seriesuid for e in train}
    val_series = {e.seriesuid for e in val}
    test_series = {e.seriesuid for e in test}

    assert train_series.isdisjoint(val_series)
    assert train_series.isdisjoint(test_series)
    assert val_series.isdisjoint(test_series)
    assert len(train) + len(val) + len(test) == 10


def test_class_distribution_counts_nodule_and_non_nodule():
    entries = [
        CandidateEntry(seriesuid="s1", coord_xyz=(0, 0, 0), label=0),
        CandidateEntry(seriesuid="s1", coord_xyz=(1, 1, 1), label=0),
        CandidateEntry(seriesuid="s2", coord_xyz=(2, 2, 2), label=1),
    ]
    assert class_distribution(entries) == {"non_nodule": 2, "nodule": 1}


def test_extract_candidate_patch_has_requested_shape_and_unit_range(tmp_path):
    path = _write_synthetic_mhd(tmp_path, "series-a", size=(60, 60, 60))
    image = sitk.ReadImage(str(path))
    crop_size = (32, 48, 48)

    patch = extract_candidate_patch(image, coord_xyz=(30.0, 30.0, 30.0), crop_size=crop_size)

    assert patch.shape == crop_size
    assert patch.dtype == np.float32
    assert patch.min() >= 0.0 and patch.max() <= 1.0


def test_extract_candidate_patch_pads_near_volume_edge(tmp_path):
    """A candidate near the volume boundary must still yield a
    full-size patch (air-padded), not crash or return a smaller array —
    real LUNA16 candidates do sit close to scan edges."""
    path = _write_synthetic_mhd(tmp_path, "series-a", size=(20, 20, 20))
    image = sitk.ReadImage(str(path))
    crop_size = (32, 48, 48)

    patch = extract_candidate_patch(image, coord_xyz=(1.0, 1.0, 1.0), crop_size=crop_size)

    assert patch.shape == crop_size
    assert np.isfinite(patch).all()


def test_ct_candidate_dataset_returns_channel_first_patch_and_label(tmp_path):
    _write_synthetic_mhd(tmp_path, "series-a", size=(60, 60, 60))
    entries = [CandidateEntry(seriesuid="series-a", coord_xyz=(30.0, 30.0, 30.0), label=1)]
    crop_size = (32, 48, 48)

    dataset = CTCandidateDataset(entries, dataset_root=tmp_path, crop_size=crop_size)

    assert len(dataset) == 1
    patch, label = dataset[0]
    assert patch.shape == (1, *crop_size)  # channel dim added
    assert label == 1


def test_ct_candidate_dataset_caches_repeated_series_avoids_redundant_reloads(
    tmp_path, monkeypatch
):
    """Real bug this guards against: reloading + resampling the full
    volume for every candidate is ~1000x more work than doing it once
    per series (measured against real subset0 data — see PROGRESS_LOG.md
    Day 7 entry). 5 candidates from the SAME series must trigger exactly
    1 real volume load, not 5."""
    _write_synthetic_mhd(tmp_path, "series-a", size=(60, 60, 60))
    entries = [
        CandidateEntry(seriesuid="series-a", coord_xyz=(float(c), float(c), float(c)), label=0)
        for c in (10, 15, 20, 25, 30)
    ]

    load_count = 0
    real_load_ct_volume = dataset_3d.load_ct_volume

    def counting_load_ct_volume(path):
        nonlocal load_count
        load_count += 1
        return real_load_ct_volume(path)

    monkeypatch.setattr(dataset_3d, "load_ct_volume", counting_load_ct_volume)

    dataset = CTCandidateDataset(entries, dataset_root=tmp_path, crop_size=(8, 8, 8))
    for i in range(len(dataset)):
        dataset[i]

    assert load_count == 1


def test_ct_candidate_dataset_respects_cache_size_eviction(tmp_path, monkeypatch):
    """With cache_size=1, alternating between two series can't keep both
    resident -- every access must reload, proving eviction actually
    happens rather than the cache silently growing unbounded."""
    _write_synthetic_mhd(tmp_path, "series-a", size=(60, 60, 60))
    _write_synthetic_mhd(tmp_path, "series-b", size=(60, 60, 60))
    entries = [
        CandidateEntry(seriesuid="series-a", coord_xyz=(30.0, 30.0, 30.0), label=0),
        CandidateEntry(seriesuid="series-b", coord_xyz=(30.0, 30.0, 30.0), label=0),
        CandidateEntry(seriesuid="series-a", coord_xyz=(30.0, 30.0, 30.0), label=0),
        CandidateEntry(seriesuid="series-b", coord_xyz=(30.0, 30.0, 30.0), label=0),
    ]

    load_count = 0
    real_load_ct_volume = dataset_3d.load_ct_volume

    def counting_load_ct_volume(path):
        nonlocal load_count
        load_count += 1
        return real_load_ct_volume(path)

    monkeypatch.setattr(dataset_3d, "load_ct_volume", counting_load_ct_volume)

    dataset = CTCandidateDataset(entries, dataset_root=tmp_path, crop_size=(8, 8, 8), cache_size=1)
    for i in range(len(dataset)):
        dataset[i]

    assert load_count == 4  # every access misses the 1-entry cache


def test_ct_candidate_dataset_default_cache_survives_random_interleaved_access(
    tmp_path, monkeypatch
):
    """Regression test for the real bug this caught: training accesses
    candidates in whatever order WeightedRandomSampler shuffles them in
    (effectively random across the whole split), NOT grouped by series.
    A small fixed cache_size verified only against sequential same-series
    access looked fast but was actually no better than uncached once
    real (interleaved) access order was tested — see PROGRESS_LOG.md Day
    7 entry. The default (cache_size=None -> auto-sized to the number of
    distinct series in `entries`) must survive this: each of N distinct
    series should trigger exactly one real load, no matter how their
    candidates are interleaved."""
    n_series = 5
    per_series = 4
    for i in range(n_series):
        _write_synthetic_mhd(tmp_path, f"series-{i}", size=(60, 60, 60))

    # Interleaved, NOT grouped by series: series-0, series-1, ..., series-4,
    # series-0, series-1, ... -- the exact pattern that broke a small
    # fixed-size LRU cache.
    entries = []
    for _ in range(per_series):
        for i in range(n_series):
            entries.append(
                CandidateEntry(seriesuid=f"series-{i}", coord_xyz=(30.0, 30.0, 30.0), label=0)
            )

    load_count = 0
    real_load_ct_volume = dataset_3d.load_ct_volume

    def counting_load_ct_volume(path):
        nonlocal load_count
        load_count += 1
        return real_load_ct_volume(path)

    monkeypatch.setattr(dataset_3d, "load_ct_volume", counting_load_ct_volume)

    dataset = CTCandidateDataset(entries, dataset_root=tmp_path, crop_size=(8, 8, 8))
    assert dataset.cache_size == n_series  # auto-sized correctly
    for i in range(len(dataset)):
        dataset[i]

    assert load_count == n_series  # each series loaded exactly once, despite interleaving


def test_ct_candidate_dataset_cache_produces_same_result_as_uncached(tmp_path):
    """The cached path must return numerically identical output to the
    original one-shot extract_candidate_patch -- caching is purely a
    performance change, not a behavior change."""
    _write_synthetic_mhd(tmp_path, "series-a", size=(60, 60, 60))
    coord = (30.0, 30.0, 30.0)
    crop_size = (16, 16, 16)

    image = dataset_3d.hu_convert(sitk.ReadImage(str(tmp_path / "series-a.mhd")))
    expected = extract_candidate_patch(image, coord, crop_size=crop_size)

    dataset = CTCandidateDataset(
        [CandidateEntry(seriesuid="series-a", coord_xyz=coord, label=0)],
        dataset_root=tmp_path,
        crop_size=crop_size,
    )
    patch, _label = dataset[0]

    np.testing.assert_array_equal(patch[0], expected)
