"""Tests for the Day 11 CT slice-preview + pixel-to-world conversion
(src/vision/ct_preview.py). Synthetic in-memory volumes only, same
convention as tests/test_preprocess_3d.py.
"""
import numpy as np
import pytest
import SimpleITK as sitk

from src.preprocessing.preprocess_3d import InvalidCTError
from src.vision.ct_preview import (
    InvalidSliceError,
    extract_display_slice,
    pixel_to_world,
    render_candidate_preview_png,
    world_to_pixel,
)


def _write_synthetic_mhd(tmp_path, size=(30, 30, 12), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(seed=0)
    array = rng.integers(-1000, 400, size=(size[2], size[1], size[0])).astype(np.int16)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    path = tmp_path / "synthetic_scan.mhd"
    sitk.WriteImage(image, str(path))
    return path


def test_extract_display_slice_missing_file_raises(tmp_path):
    with pytest.raises(InvalidCTError):
        extract_display_slice(tmp_path / "does_not_exist.mhd")


def test_extract_display_slice_defaults_to_middle_slice(tmp_path):
    path = _write_synthetic_mhd(tmp_path, size=(30, 30, 12))
    result = extract_display_slice(path)
    assert result.num_slices == 12
    assert result.slice_index == 6  # 12 // 2


def test_extract_display_slice_respects_explicit_slice_index(tmp_path):
    path = _write_synthetic_mhd(tmp_path, size=(30, 30, 12))
    result = extract_display_slice(path, slice_index=3)
    assert result.slice_index == 3


def test_extract_display_slice_out_of_range_raises(tmp_path):
    path = _write_synthetic_mhd(tmp_path, size=(30, 30, 12))
    with pytest.raises(InvalidSliceError):
        extract_display_slice(path, slice_index=999)
    with pytest.raises(InvalidSliceError):
        extract_display_slice(path, slice_index=-1)


def test_extract_display_slice_output_is_uint8_and_matches_dimensions(tmp_path):
    path = _write_synthetic_mhd(tmp_path, size=(30, 30, 12), spacing=(1.0, 1.0, 1.0))
    result = extract_display_slice(path)
    assert result.image_array.dtype == np.uint8
    assert result.image_array.shape == (result.height, result.width)
    assert result.image_array.min() >= 0 and result.image_array.max() <= 255


def test_extract_display_slice_returns_real_origin_and_spacing(tmp_path):
    path = _write_synthetic_mhd(
        tmp_path, size=(30, 30, 12), spacing=(1.0, 1.0, 1.0), origin=(-10.0, -20.0, -30.0)
    )
    result = extract_display_slice(path)
    assert result.origin == pytest.approx((-10.0, -20.0, -30.0))
    assert result.spacing == pytest.approx((1.0, 1.0, 1.0))


def test_pixel_to_world_applies_origin_and_spacing():
    world = pixel_to_world(
        row=5, col=10, slice_index=2, origin=(-100.0, -50.0, -30.0), spacing=(1.0, 1.0, 1.0)
    )
    assert world == pytest.approx((-90.0, -45.0, -28.0))


def test_pixel_to_world_round_trips_a_real_extracted_slice(tmp_path):
    """Pick a real voxel from a real (synthetic) extracted slice, and
    confirm converting its pixel position back to world coordinates,
    then forward again via the same transform SimpleITK uses, recovers
    the same point -- the actual property that matters for the
    click-to-pick UI."""
    path = _write_synthetic_mhd(
        tmp_path, size=(30, 30, 12), spacing=(1.0, 1.0, 1.0), origin=(5.0, 5.0, 5.0)
    )
    result = extract_display_slice(path, slice_index=7)

    row, col = 15, 20
    world = pixel_to_world(row, col, result.slice_index, result.origin, result.spacing)

    # Forward-check via SimpleITK's own transform (same one dataset_3d.py's
    # crop_patch_from_resampled uses) -- must land back on (col, row, slice).
    image = sitk.ReadImage(str(path))
    ix, iy, iz = image.TransformPhysicalPointToContinuousIndex(world)
    assert round(ix) == col
    assert round(iy) == row
    assert round(iz) == result.slice_index


def test_world_to_pixel_is_the_exact_inverse_of_pixel_to_world():
    origin = (-100.0, -50.0, -30.0)
    spacing = (1.0, 1.0, 1.0)
    row, col, slice_index = 5, 10, 2

    world = pixel_to_world(row, col, slice_index, origin, spacing)
    round_tripped = world_to_pixel(world, origin, spacing)

    assert round_tripped == (row, col, slice_index)


def test_world_to_pixel_round_trips_a_real_candidate_coordinate(tmp_path):
    """Day 11.1: the marked-slice report preview relies on this to land
    the marker exactly where a candidate was actually analyzed -- same
    round-trip property test_pixel_to_world_round_trips_a_real_extracted_slice
    proves for the forward direction, checked here for the inverse."""
    path = _write_synthetic_mhd(
        tmp_path, size=(30, 30, 12), spacing=(1.0, 1.0, 1.0), origin=(5.0, 5.0, 5.0)
    )
    result = extract_display_slice(path, slice_index=7)
    expected_row, expected_col = 15, 20
    world = pixel_to_world(expected_row, expected_col, result.slice_index, result.origin, result.spacing)

    row, col, slice_index = world_to_pixel(world, result.origin, result.spacing)

    assert (row, col, slice_index) == (expected_row, expected_col, result.slice_index)


def test_render_candidate_preview_png_returns_a_real_marked_image(tmp_path):
    path = _write_synthetic_mhd(tmp_path, size=(30, 30, 12), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    # A real coordinate inside the volume, in the same world-mm space
    # extract_candidate_patch/analyze_ct expect.
    coord_xyz = (10.0, 10.0, 6.0)

    png_bytes = render_candidate_preview_png(path, coord_xyz)

    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # real PNG signature, not empty/garbage bytes
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(png_bytes))
    assert img.mode == "RGB"  # marker is drawn in color, not grayscale
    assert img.size == (30, 30)


def test_render_candidate_preview_png_clamps_an_out_of_range_slice(tmp_path):
    """A candidate coordinate whose z rounds to just outside the volume
    (e.g. a saved candidate near the scan's edge) must still produce a
    preview -- clamped to the nearest real slice -- rather than crash a
    whole report generation over one image."""
    path = _write_synthetic_mhd(tmp_path, size=(30, 30, 12), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    coord_xyz = (10.0, 10.0, 999.0)  # far past the last slice

    png_bytes = render_candidate_preview_png(path, coord_xyz)

    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
