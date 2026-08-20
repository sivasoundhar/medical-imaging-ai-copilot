"""Tests for the 3D CNN model builder. Random init only (no pretrained
option exists for 3D volumes — see model_3d.py docstring)."""
import torch

from src.vision.model_3d import build_model_3d, count_trainable_parameters, log_gpu_memory


def test_output_shape_matches_num_classes():
    model = build_model_3d(num_classes=2)
    model.eval()

    dummy_input = torch.randn(2, 1, 32, 48, 48)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 2)


def test_different_num_classes_configurable():
    model = build_model_3d(num_classes=3)
    dummy_input = torch.randn(2, 1, 32, 48, 48)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 3)


def test_adaptive_pooling_handles_a_different_crop_size():
    """The head uses AdaptiveAvgPool3d, so a non-default crop size must
    still work without a shape mismatch."""
    model = build_model_3d(num_classes=2)
    dummy_input = torch.randn(1, 1, 16, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 2)


def test_all_parameters_trainable_by_default():
    model = build_model_3d(num_classes=2)
    total = sum(p.numel() for p in model.parameters())
    assert count_trainable_parameters(model) == total
    assert total > 0


def test_log_gpu_memory_is_a_noop_on_cpu(capsys):
    log_gpu_memory(torch.device("cpu"))
    captured = capsys.readouterr()
    assert captured.out == ""  # nothing printed on CPU-only devices
