"""Grad-CAM explainability for the trained 2D X-ray model (Day 5).

Pipeline (PROJECT_SPEC.md Section 51, Day 5):

    X-ray -> model_2d.pth -> prediction -> Grad-CAM -> heatmap -> overlay

This module is a read-only *inference-time* consumer of an already-trained
checkpoint (`training/checkpoints/model_2d_best.pth` from Day 4) — it does
not touch `training/train_2d.py` in any way, per the Day 5 instruction.

Grad-CAM Rule (PROJECT_SPEC.md Section 48): Grad-CAM explains model
activation/focus — which pixels most influenced the prediction. It does
NOT prove that a disease is present and is not a diagnosis. Every result
produced here carries `GRADCAM_DISCLAIMER`; any future caller (UI/API,
starting Day 8/9/10) must surface it verbatim, not paraphrase it away.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.preprocessing.preprocess_2d import DEFAULT_SIZE, load_xray, normalize, resize, to_rgb
from src.vision.dataset_2d import LABELS
from src.vision.model_2d import build_resnet50, get_device

GRADCAM_DISCLAIMER = (
    "Grad-CAM shows which regions of the image most influenced the model's "
    "prediction (model activation/focus). It does not prove that a disease "
    "is present and is not a diagnosis."
)

_IDX_TO_LABEL = {v: k for k, v in LABELS.items()}


@dataclass
class GradCAMResult:
    predicted_label: str
    probability: float  # probability of predicted_label
    class_probabilities: dict[str, float]
    heatmap: np.ndarray  # (H, W) float32 in [0, 1]
    overlay: np.ndarray  # (H, W, 3) uint8 RGB
    # The resized-to-DEFAULT_SIZE RGB image the model actually saw, before
    # the heatmap is blended in -- exposed so a UI can show "Original" next
    # to "Grad-CAM Overlay" at identical dimensions (real bug hit live:
    # showing the raw uploaded file's own aspect ratio next to the always-
    # square overlay made the two look mismatched in size, and cropping the
    # raw file to force a match cut off real image content -- e.g. the "R"
    # laterality marker). Same resolution, same aspect ratio as `overlay`,
    # by construction (both come from `image_resized`).
    resized_original: np.ndarray  # (H, W, 3) uint8 RGB
    target_layer: str
    disclaimer: str = field(default=GRADCAM_DISCLAIMER)


def load_trained_model(
    checkpoint_path: str | Path, device: torch.device | None = None
) -> torch.nn.Module:
    """Load a Day 4 checkpoint (state_dict only, see
    `training/train_2d.py:save_checkpoint`) for inference.

    `pretrained=False` here — real weights come from the checkpoint, not
    ImageNet init; loading ImageNet weights first would just be wasted
    download/compute immediately overwritten by `load_state_dict`.
    """
    device = device or get_device()
    model = build_resnet50(pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _target_layer(model: torch.nn.Module) -> torch.nn.Module:
    # Last conv block of ResNet50's final residual stage — the standard
    # Grad-CAM choice for ResNet architectures (deepest spatial features,
    # before global average pooling collapses them).
    return model.layer4[-1]


def generate_gradcam(
    image_path: str | Path,
    model: torch.nn.Module,
    device: torch.device | None = None,
) -> GradCAMResult:
    """Run prediction + Grad-CAM + overlay for one X-ray image against an
    already-loaded model (see `load_trained_model`).

    Raises `InvalidXrayError` (from `src.preprocessing.preprocess_2d`) for
    missing/corrupt/zero-area images — same validation the training
    pipeline uses, so a "valid X-ray" means the same thing in both places.
    """
    device = device or get_device()
    # Callers aren't guaranteed to have moved `model` to `device` themselves
    # (load_trained_model does, but a bare build_resnet50() model passed in
    # directly won't have) -- move it here so input/model always match.
    # No-op if it's already on the right device.
    model = model.to(device)

    image = to_rgb(load_xray(image_path))
    image_resized = resize(image, DEFAULT_SIZE)
    normalized = normalize(image_resized)  # (3, H, W) float32, ImageNet-normalized

    input_tensor = torch.from_numpy(normalized).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0]
    predicted_idx = int(torch.argmax(probs).item())
    class_probabilities = {
        _IDX_TO_LABEL[i]: float(probs[i].item()) for i in range(len(_IDX_TO_LABEL))
    }

    # GradCAM registers forward/backward hooks and needs a fresh backward
    # pass per call — it manages its own gradient context, so this runs
    # outside the `torch.no_grad()` block above.
    cam = GradCAM(model=model, target_layers=[_target_layer(model)])
    grayscale_cam = cam(
        input_tensor=input_tensor, targets=[ClassifierOutputTarget(predicted_idx)]
    )[0]  # (H, W) float32 in [0, 1]

    # Plain [0, 1] RGB (not ImageNet-normalized) — show_cam_on_image needs
    # a displayable background, not the model's normalized input.
    rgb_float = np.asarray(image_resized, dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)  # (H, W, 3) uint8

    return GradCAMResult(
        predicted_label=_IDX_TO_LABEL[predicted_idx],
        probability=class_probabilities[_IDX_TO_LABEL[predicted_idx]],
        class_probabilities=class_probabilities,
        heatmap=grayscale_cam,
        overlay=overlay,
        resized_original=np.asarray(image_resized, dtype=np.uint8),
        target_layer="layer4[-1]",
    )


def save_overlay(result: GradCAMResult, output_dir: str | Path, filename_stem: str) -> Path:
    """Save the overlay as a PNG and return its path — the "heatmap
    path/reference" required in the Day 5 API output shape."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename_stem}_gradcam.png"
    Image.fromarray(result.overlay).save(path)
    return path


def save_resized_original(result: GradCAMResult, output_dir: str | Path, filename_stem: str) -> Path:
    """Save the resized-to-DEFAULT_SIZE original (pre-overlay) as a PNG,
    same convention as `save_overlay` -- lets a UI pair "Original" with
    "Grad-CAM Overlay" at identical dimensions (see `GradCAMResult.
    resized_original`'s docstring for why)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename_stem}_original.png"
    Image.fromarray(result.resized_original).save(path)
    return path


def explain_xray(
    image_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path = "outputs/gradcam",
    device: torch.device | None = None,
) -> dict:
    """End-to-end Day 5 pipeline: load model -> predict -> Grad-CAM ->
    overlay -> save heatmap -> structured metadata.

    Returns the shape PROJECT_SPEC.md's Day 5 "API output" section
    specifies: prediction, probability, heatmap path/reference, model
    metadata, disclaimer. Not wired to a live API endpoint yet — routers
    start Day 8/9/10 (see `src/main.py`) — this is the service-layer
    function those endpoints will call directly.
    """
    device = device or get_device()
    model = load_trained_model(checkpoint_path, device=device)
    result = generate_gradcam(image_path, model, device=device)
    heatmap_path = save_overlay(result, output_dir, Path(image_path).stem)
    resized_original_path = save_resized_original(result, output_dir, Path(image_path).stem)

    return {
        "prediction": result.predicted_label,
        "probability": result.probability,
        "class_probabilities": result.class_probabilities,
        "heatmap_path": str(heatmap_path),
        "resized_original_path": str(resized_original_path),
        "model_metadata": {
            "architecture": "resnet50",
            "checkpoint": Path(checkpoint_path).name,
            "target_layer": result.target_layer,
        },
        "disclaimer": GRADCAM_DISCLAIMER,
    }
