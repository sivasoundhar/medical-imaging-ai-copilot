"""Tests for CT preprocessing. Uses synthetic in-memory volumes only —
never real patient data."""
import numpy as np
import pytest
import SimpleITK as sitk

from src.preprocessing.preprocess_3d import (
    InvalidCTError,
    apply_hu_window,
    extract_metadata,
    load_ct_volume,
    preprocess_ct,
)


def _write_synthetic_mhd(tmp_path, size=(20, 20, 10), spacing=(0.7, 0.7, 2.5)):
    """Create a small synthetic HU-range volume and write it as .mhd/.raw,
    matching the LUNA16 on-disk format."""
    rng = np.random.default_rng(seed=0)
    # (D, H, W) order for SimpleITK array constructor
    array = rng.integers(-1000, 400, size=(size[2], size[1], size[0])).astype(np.int16)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    path = tmp_path / "synthetic_scan.mhd"
    sitk.WriteImage(image, str(path))
    return path


def test_missing_file_raises(tmp_path) -> None:
    with pytest.raises(InvalidCTError):
        load_ct_volume(tmp_path / "does_not_exist.mhd")


def test_corrupt_file_raises(tmp_path) -> None:
    bad_file = tmp_path / "corrupt.mhd"
    bad_file.write_text("not a real metaimage header")
    with pytest.raises(InvalidCTError):
        load_ct_volume(bad_file)


def test_valid_ct_volume_loads_and_preprocesses(tmp_path) -> None:
    path = _write_synthetic_mhd(tmp_path, size=(20, 20, 10), spacing=(0.7, 0.7, 2.5))

    array, metadata = preprocess_ct(path)

    assert array.ndim == 3
    assert np.isfinite(array).all()
    assert array.min() >= 0.0 and array.max() <= 1.0
    assert metadata.size == (20, 20, 10)
    assert metadata.spacing == pytest.approx((0.7, 0.7, 2.5))


def test_metadata_extraction(tmp_path) -> None:
    path = _write_synthetic_mhd(tmp_path, size=(16, 16, 8), spacing=(1.0, 1.0, 1.0))
    image = load_ct_volume(path)
    metadata = extract_metadata(image)

    assert metadata.size == (16, 16, 8)
    assert len(metadata.origin) == 3
    assert len(metadata.direction) == 9  # 3x3 direction cosine matrix, flattened


def test_hu_window_clips_and_scales_to_unit_range() -> None:
    array = np.array([-2000, -1000, -300, 400, 2000], dtype=np.float32)
    windowed = apply_hu_window(array, window=(-1000, 400))

    assert windowed.min() >= 0.0
    assert windowed.max() <= 1.0
    # values below the window floor and above the ceiling both clip to the edges
    assert windowed[0] == windowed[1] == 0.0
    assert windowed[3] == windowed[4] == 1.0
