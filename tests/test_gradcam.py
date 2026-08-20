"""Tests for Day 5 Grad-CAM explainability (src/vision/gradcam.py).

Synthetic data only, per project convention (see test_dataset_2d.py /
test_model_2d.py) — no real patient data, and no dependency on the real
Day 4 checkpoint (94MB, gitignored, may not exist in a fresh clone/CI).
Models here are freshly built with `pretrained=False` (random init) —
fast, deterministic-shape, and never triggers an ImageNet download.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.preprocessing.preprocess_2d import DEFAULT_SIZE, InvalidXrayError
from src.vision.gradcam import (
    GRADCAM_DISCLAIMER,
    explain_xray,
    generate_gradcam,
    load_trained_model,
    save_overlay,
    save_resized_original,
)
from src.vision.model_2d import build_resnet50


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = build_resnet50(pretrained=False)
    m.eval()
    return m


def _make_synthetic_xray(path: Path, size: tuple[int, int] = (400, 300)) -> Path:
    """A random-noise grayscale image saved as JPEG — mimics a real
    chest X-ray file closely enough to exercise the full preprocessing +
    Grad-CAM pipeline without any real patient data."""
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(size[1], size[0]), dtype=np.uint8)
    image = Image.fromarray(array, mode="L")
    image.save(path)
    return path


def test_generate_gradcam_on_valid_image_returns_expected_shapes(tmp_path, model):
    image_path = _make_synthetic_xray(tmp_path / "xray.jpeg")

    result = generate_gradcam(image_path, model)

    assert result.heatmap.shape == DEFAULT_SIZE
    assert result.overlay.shape == (*DEFAULT_SIZE, 3)
    assert result.overlay.dtype == np.uint8
    # Real bug this guards against: the UI showed the original next to
    # Grad-CAM at mismatched sizes because it used the raw uploaded file
    # instead of this resized-to-DEFAULT_SIZE version, which is
    # guaranteed identical dimensions to `overlay` by construction (both
    # derive from the same `image_resized`).
    assert result.resized_original.shape == (*DEFAULT_SIZE, 3)
    assert result.resized_original.dtype == np.uint8


def test_generate_gradcam_output_dimensions_match_source_regardless_of_input_size(
    tmp_path, model
):
    # A differently-sized source image should still produce output at
    # DEFAULT_SIZE — preprocessing resizes before the model ever sees it.
    image_path = _make_synthetic_xray(tmp_path / "xray_odd_size.jpeg", size=(1024, 768))

    result = generate_gradcam(image_path, model)

    assert result.heatmap.shape == DEFAULT_SIZE
    assert result.overlay.shape == (*DEFAULT_SIZE, 3)


def test_heatmap_values_within_unit_range(tmp_path, model):
    image_path = _make_synthetic_xray(tmp_path / "xray.jpeg")

    result = generate_gradcam(image_path, model)

    assert result.heatmap.min() >= 0.0
    assert result.heatmap.max() <= 1.0 + 1e-6


def test_prediction_and_class_probabilities_are_consistent(tmp_path, model):
    image_path = _make_synthetic_xray(tmp_path / "xray.jpeg")

    result = generate_gradcam(image_path, model)

    assert result.predicted_label in {"NORMAL", "PNEUMONIA"}
    assert 0.0 <= result.probability <= 1.0
    assert set(result.class_probabilities) == {"NORMAL", "PNEUMONIA"}
    assert result.class_probabilities[result.predicted_label] == pytest.approx(
        result.probability
    )
    assert sum(result.class_probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert result.predicted_label == max(
        result.class_probabilities, key=result.class_probabilities.get
    )


def test_disclaimer_states_activation_not_proof_of_disease(tmp_path, model):
    image_path = _make_synthetic_xray(tmp_path / "xray.jpeg")

    result = generate_gradcam(image_path, model)

    assert result.disclaimer == GRADCAM_DISCLAIMER
    assert "does not prove" in result.disclaimer
    assert "not a diagnosis" in result.disclaimer


def test_generate_gradcam_on_missing_file_raises_invalid_xray_error(tmp_path, model):
    missing_path = tmp_path / "does_not_exist.jpeg"

    with pytest.raises(InvalidXrayError):
        generate_gradcam(missing_path, model)


def test_generate_gradcam_on_corrupt_file_raises_invalid_xray_error(tmp_path, model):
    corrupt_path = tmp_path / "corrupt.jpeg"
    corrupt_path.write_bytes(b"not actually an image")

    with pytest.raises(InvalidXrayError):
        generate_gradcam(corrupt_path, model)


def test_save_overlay_writes_readable_png(tmp_path, model):
    image_path = _make_synthetic_xray(tmp_path / "xray.jpeg")
    result = generate_gradcam(image_path, model)

    out_path = save_overlay(result, tmp_path / "out", "xray")

    assert out_path.exists()
    saved = Image.open(out_path)
    assert saved.size == DEFAULT_SIZE


def test_save_resized_original_writes_a_png_matching_the_overlays_dimensions(tmp_path, model):
    image_path = _make_synthetic_xray(tmp_path / "xray.jpeg")
    result = generate_gradcam(image_path, model)

    overlay_path = save_overlay(result, tmp_path / "out", "xray")
    original_path = save_resized_original(result, tmp_path / "out", "xray")

    assert original_path.exists()
    assert original_path != overlay_path  # distinct files, not accidentally overwriting each other
    saved = Image.open(original_path)
    assert saved.size == DEFAULT_SIZE
    assert saved.size == Image.open(overlay_path).size


def test_load_trained_model_restores_checkpoint_weights_and_eval_mode(tmp_path, model):
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)

    loaded = load_trained_model(checkpoint_path, device=torch.device("cpu"))

    assert not loaded.training  # eval() mode, no accidental dropout/BN-update at inference
    # .cpu() both sides: `model` is the module-scoped fixture, which earlier
    # tests in this file may have moved onto CUDA via generate_gradcam() --
    # this test cares about value equality, not which device either lives on.
    for p_orig, p_loaded in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(p_orig.cpu(), p_loaded.cpu())


def test_explain_xray_end_to_end_returns_day5_api_output_shape(tmp_path, model):
    image_path = _make_synthetic_xray(tmp_path / "xray.jpeg")
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)
    output_dir = tmp_path / "gradcam_out"

    output = explain_xray(
        image_path, checkpoint_path, output_dir=output_dir, device=torch.device("cpu")
    )

    # PROJECT_SPEC.md Day 5 "API output": prediction, probability, heatmap
    # path/reference, model metadata, disclaimer.
    assert output["prediction"] in {"NORMAL", "PNEUMONIA"}
    assert 0.0 <= output["probability"] <= 1.0
    assert Path(output["heatmap_path"]).exists()
    assert Path(output["resized_original_path"]).exists()
    assert output["model_metadata"]["architecture"] == "resnet50"
    assert output["model_metadata"]["checkpoint"] == "checkpoint.pth"
    assert output["disclaimer"] == GRADCAM_DISCLAIMER
