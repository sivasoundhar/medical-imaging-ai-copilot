"""3D CNN for LUNA16 nodule-candidate classification (binary: nodule vs
non-nodule — see `dataset_3d.py`'s `CandidateEntry.label`).

Unlike `model_2d.py`'s ResNet50, this is trained from scratch — there is
no widely-available ImageNet-scale pretrained 3D-volume backbone to
transfer-learn from. Architecture is a compact from-scratch 3D CNN
(Conv3d/BatchNorm3d/ReLU/MaxPool3d blocks -> global average pool -> FC),
sized for the compute budget in PROJECT_SPEC.md Section 33 (Colab, small
batches, small crops) — a candidate-classification/localization model,
not a research-grade or clinical-grade segmentation network.
"""
import torch
from torch import nn

NUM_CLASSES = 2  # non-nodule, nodule -- see src/vision/dataset_3d.py CandidateEntry.label


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class Nodule3DCNN(nn.Module):
    """Input: (N, in_channels, D, H, W) — default crop size (32, 48, 48)
    per `dataset_3d.CROP_SIZE_VOXELS`, but any size works since the head
    uses adaptive pooling. Output: (N, num_classes) logits."""

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 16),
            nn.MaxPool3d(2),
            _conv_block(16, 32),
            nn.MaxPool3d(2),
            _conv_block(32, 64),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model_3d(num_classes: int = NUM_CLASSES, in_channels: int = 1) -> nn.Module:
    return Nodule3DCNN(num_classes=num_classes, in_channels=in_channels)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_gpu_memory(device: torch.device, prefix: str = "") -> None:
    """Best-effort GPU memory logging (spec's Day 6 "log GPU memory where
    practical"). No-op on CPU-only devices — nothing to report."""
    if device.type != "cuda":
        return
    allocated_mb = torch.cuda.memory_allocated(device) / (1024**2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024**2)
    print(f"{prefix}GPU memory: allocated={allocated_mb:.1f}MB reserved={reserved_mb:.1f}MB")
