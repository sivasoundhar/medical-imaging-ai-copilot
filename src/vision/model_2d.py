"""ResNet50 transfer-learning model for X-ray classification.

Baseline configuration per PROJECT_SPEC.md Section 45: pretrained
ResNet50, binary classification head (NORMAL vs PNEUMONIA for the
current Kaggle dataset — see dataset_2d.py LABELS).
"""
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

NUM_CLASSES = 2  # NORMAL, PNEUMONIA — see src/vision/dataset_2d.py LABELS


def build_resnet50(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a ResNet50 with its final FC layer replaced for
    `num_classes`-way classification.

    freeze_backbone=True freezes every layer except the new classifier
    head — a lighter-weight transfer-learning mode useful for quick
    smoke tests or very small datasets.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # always trainable, even if frozen above

    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
