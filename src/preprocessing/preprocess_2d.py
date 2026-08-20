"""X-ray (2D) preprocessing.

Loads a chest X-ray image, validates it, and converts it into a
normalized tensor ready for the ResNet50 pipeline (Day 3). No model
code lives here — this module only prepares input data.
"""
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

# ImageNet normalization stats — ResNet50 is pretrained on these.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_SIZE = (224, 224)


class InvalidXrayError(ValueError):
    """Raised when an X-ray image fails loading or validation."""


def load_xray(path: str | Path) -> Image.Image:
    """Load an image file and validate it is a usable X-ray image.

    Raises InvalidXrayError for missing files, unreadable/corrupt files,
    or images with degenerate (zero-area) dimensions.
    """
    path = Path(path)
    if not path.exists():
        raise InvalidXrayError(f"File not found: {path}")

    try:
        image = Image.open(path)
        image.load()  # force decode now, not lazily later
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidXrayError(f"Could not read image file: {path}") from exc

    if image.width == 0 or image.height == 0:
        raise InvalidXrayError(f"Image has zero-area dimensions: {path}")

    return image


def to_rgb(image: Image.Image) -> Image.Image:
    """Chest X-rays are commonly stored grayscale ('L') or with an alpha
    channel; ResNet50 expects 3-channel RGB input."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def resize(image: Image.Image, size: tuple[int, int] = DEFAULT_SIZE) -> Image.Image:
    return image.resize(size, Image.Resampling.BILINEAR)


def normalize(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to a normalized (C, H, W) float32 array."""
    array = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, C) in [0, 1]
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return array.transpose(2, 0, 1)  # -> (C, H, W)


def preprocess_xray(path: str | Path, size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """Full pipeline: load -> validate -> RGB -> resize -> normalize.

    Returns a (3, H, W) float32 array. Tensor conversion for a specific
    framework (torch.from_numpy, etc.) happens at the DataLoader layer
    in Day 3, not here — this stays framework-agnostic.
    """
    image = load_xray(path)
    image = to_rgb(image)
    image = resize(image, size)
    return normalize(image)
