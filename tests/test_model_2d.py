"""Tests for the ResNet50 model builder. pretrained=False everywhere here
so tests never depend on downloading ImageNet weights."""
import torch

from src.vision.model_2d import build_resnet50, count_trainable_parameters


def test_output_shape_matches_num_classes() -> None:
    model = build_resnet50(num_classes=2, pretrained=False)
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (4, 2)


def test_freeze_backbone_only_trains_head() -> None:
    frozen_model = build_resnet50(num_classes=2, pretrained=False, freeze_backbone=True)
    full_model = build_resnet50(num_classes=2, pretrained=False, freeze_backbone=False)

    frozen_trainable = count_trainable_parameters(frozen_model)
    full_trainable = count_trainable_parameters(full_model)

    # frozen model should only train the final FC layer (in_features*2 + 2 bias terms)
    fc_params = sum(p.numel() for p in frozen_model.fc.parameters())
    assert frozen_trainable == fc_params
    assert frozen_trainable < full_trainable


def test_different_num_classes_configurable() -> None:
    model = build_resnet50(num_classes=5, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 5)
