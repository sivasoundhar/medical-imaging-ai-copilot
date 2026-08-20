"""Tests for X-ray preprocessing. Uses synthetic in-memory images only —
never real patient data."""
import numpy as np
import pytest
from PIL import Image

from src.preprocessing.preprocess_2d import (
    DEFAULT_SIZE,
    InvalidXrayError,
    normalize,
    preprocess_xray,
    resize,
    to_rgb,
)


def _make_synthetic_image(path, size=(300, 250), mode="L") -> None:
    array = np.random.default_rng(seed=0).integers(0, 255, size=size[::-1], dtype=np.uint8)
    Image.fromarray(array, mode=mode).save(path)


def test_missing_file_raises(tmp_path) -> None:
    with pytest.raises(InvalidXrayError):
        preprocess_xray(tmp_path / "does_not_exist.jpeg")


def test_corrupt_file_raises(tmp_path) -> None:
    bad_file = tmp_path / "corrupt.jpeg"
    bad_file.write_bytes(b"this is not a valid jpeg")
    with pytest.raises(InvalidXrayError):
        preprocess_xray(bad_file)


def test_valid_grayscale_xray_preprocesses_to_expected_shape(tmp_path) -> None:
    path = tmp_path / "synthetic_xray.jpeg"
    _make_synthetic_image(path, size=(300, 250), mode="L")

    result = preprocess_xray(path)

    assert result.shape == (3, *DEFAULT_SIZE)
    assert result.dtype == np.float32


def test_unexpected_dimensions_still_resize_correctly(tmp_path) -> None:
    """Non-square, oddly-sized input must still normalize to DEFAULT_SIZE."""
    path = tmp_path / "odd_size.jpeg"
    _make_synthetic_image(path, size=(97, 53), mode="L")

    result = preprocess_xray(path)

    assert result.shape == (3, *DEFAULT_SIZE)


def test_to_rgb_converts_grayscale() -> None:
    grayscale = Image.new("L", (10, 10))
    rgb = to_rgb(grayscale)
    assert rgb.mode == "RGB"


def test_resize_produces_requested_size() -> None:
    image = Image.new("RGB", (500, 400))
    resized = resize(image, size=(224, 224))
    assert resized.size == (224, 224)


def test_normalize_output_is_finite() -> None:
    image = Image.new("RGB", (10, 10), color=(128, 128, 128))
    array = normalize(image)
    assert np.isfinite(array).all()
    assert array.shape == (3, 10, 10)
