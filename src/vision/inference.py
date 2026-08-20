"""Unified inference glue for both trained models (Day 8,
PROJECT_SPEC.md Section 29's `src/vision/inference.py`).

Wraps the existing 2D Grad-CAM pipeline (`gradcam.py`, Day 5) and adds
the equivalent single-candidate 3D CT inference path (new here) behind
one predictable shape, so `src/services/imaging_service.py` doesn't need
to know model internals.

CT scope note (deliberate, not a shortcut): `model_3d.pth` was trained
and evaluated as a candidate-patch classifier (see PROGRESS_LOG.md Day
7) — "is THIS specific x/y/z location a nodule or not" — the same
"stage 2 false-positive reduction" role real LUNA16-style pipelines use.
It was never trained to scan a whole CT volume and find candidates on
its own (that's a separate "stage 1" problem, not built here). So
`analyze_ct` requires a candidate coordinate as input rather than
scanning the volume automatically — see the click-to-pick coordinate UI
in the frontend for how a candidate location is chosen.

Both `analyze_*` functions raise `ModelUnavailableError` when the
requested checkpoint file doesn't exist on disk — callers must surface a
clean error, never fabricate a prediction from a missing/untrained model.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.preprocessing.preprocess_3d import load_ct_volume
from src.vision.dataset_3d import extract_candidate_patch
from src.vision.gradcam import GRADCAM_DISCLAIMER, explain_xray
from src.vision.model_2d import get_device
from src.vision.model_3d import build_model_3d

CT_DISCLAIMER = (
    "This model classifies a single candidate location you provide — it "
    "does not scan a full CT volume to find candidates on its own. "
    "Model activation, not a diagnosis; requires professional review."
)

_CT_IDX_TO_LABEL = {0: "non_nodule", 1: "nodule"}


class ModelUnavailableError(RuntimeError):
    """Raised when a requested checkpoint file doesn't exist on disk."""


@dataclass
class XrayAnalysis:
    prediction: str
    probability: float
    class_probabilities: dict[str, float]
    heatmap_path: str
    resized_original_path: str
    model_metadata: dict
    disclaimer: str


@dataclass
class CTAnalysis:
    prediction: str
    probability: float
    class_probabilities: dict[str, float]
    coord_xyz: tuple[float, float, float]
    model_metadata: dict
    disclaimer: str


def analyze_xray(image_path: str | Path, checkpoint_path: str | Path) -> XrayAnalysis:
    """Raises `InvalidXrayError` (from `src.preprocessing.preprocess_2d`,
    propagated via `explain_xray`) for a missing/corrupt image."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise ModelUnavailableError(f"2D model checkpoint not found: {checkpoint_path}")

    result = explain_xray(image_path, checkpoint_path)
    return XrayAnalysis(
        prediction=result["prediction"],
        probability=result["probability"],
        class_probabilities=result["class_probabilities"],
        heatmap_path=result["heatmap_path"],
        resized_original_path=result["resized_original_path"],
        model_metadata=result["model_metadata"],
        disclaimer=result["disclaimer"],
    )


def analyze_ct(
    volume_path: str | Path,
    coord_xyz: tuple[float, float, float],
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> CTAnalysis:
    """Raises `InvalidCTError` (from `src.preprocessing.preprocess_3d`)
    for a missing/corrupt volume."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise ModelUnavailableError(f"3D model checkpoint not found: {checkpoint_path}")

    device = device or get_device()
    image = load_ct_volume(volume_path)
    patch = extract_candidate_patch(image, coord_xyz)  # (D, H, W) float32 in [0, 1]

    model = build_model_3d().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    input_tensor = torch.from_numpy(patch[np.newaxis, np.newaxis, ...]).to(
        device, dtype=torch.float32
    )
    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0]
    predicted_idx = int(torch.argmax(probs).item())
    class_probabilities = {
        _CT_IDX_TO_LABEL[i]: float(probs[i].item()) for i in range(len(_CT_IDX_TO_LABEL))
    }

    return CTAnalysis(
        prediction=_CT_IDX_TO_LABEL[predicted_idx],
        probability=class_probabilities[_CT_IDX_TO_LABEL[predicted_idx]],
        class_probabilities=class_probabilities,
        coord_xyz=coord_xyz,
        model_metadata={"architecture": "nodule_3d_cnn", "checkpoint": checkpoint_path.name},
        disclaimer=CT_DISCLAIMER,
    )


__all__ = [
    "GRADCAM_DISCLAIMER",
    "CT_DISCLAIMER",
    "ModelUnavailableError",
    "XrayAnalysis",
    "CTAnalysis",
    "analyze_xray",
    "analyze_ct",
]
