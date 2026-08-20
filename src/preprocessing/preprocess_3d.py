"""CT (3D) preprocessing.

The LUNA16 dataset ships volumes as MetaImage pairs (`.mhd` header +
`.raw` data), read via SimpleITK — the values are already stored in
Hounsfield Units (HU), unlike raw DICOM pixel data which needs
RescaleSlope/RescaleIntercept applied first. `hu_convert()` below is a
no-op passthrough for `.mhd` input and exists so the same pipeline shape
still works if raw DICOM input is added later.

No model code lives here — this module only prepares input volumes for
the 3D CNN pipeline (Day 6).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Lung window, standard for chest CT nodule visibility (HU).
DEFAULT_HU_WINDOW = (-1000, 400)
DEFAULT_SPACING_MM = (1.0, 1.0, 1.0)


class InvalidCTError(ValueError):
    """Raised when a CT volume fails loading or validation."""


@dataclass
class CTMetadata:
    size: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]


def load_ct_volume(path: str | Path) -> sitk.Image:
    """Load a `.mhd`/`.raw` CT volume.

    Raises InvalidCTError for a missing file, a file SimpleITK cannot
    parse, or a volume with an empty (zero-voxel) dimension.
    """
    path = Path(path)
    if not path.exists():
        raise InvalidCTError(f"File not found: {path}")

    try:
        image = sitk.ReadImage(str(path))
    except RuntimeError as exc:
        raise InvalidCTError(f"Could not read CT volume: {path}") from exc

    if any(dim == 0 for dim in image.GetSize()):
        raise InvalidCTError(f"CT volume has an empty dimension: {path}")

    return image


def extract_metadata(image: sitk.Image) -> CTMetadata:
    return CTMetadata(
        size=image.GetSize(),
        spacing=image.GetSpacing(),
        origin=image.GetOrigin(),
        direction=image.GetDirection(),
    )


def hu_convert(image: sitk.Image) -> sitk.Image:
    """Ensure the volume is in HU. `.mhd`/`.raw` LUNA16 volumes are
    already HU, so this is currently a passthrough — kept as an explicit
    step so a future raw-DICOM loader can slot RescaleSlope/Intercept
    application in here without changing the rest of the pipeline."""
    return image


def resample_isotropic(
    image: sitk.Image, target_spacing_mm: tuple[float, float, float] = DEFAULT_SPACING_MM
) -> sitk.Image:
    """Resample to isotropic voxel spacing so downstream 3D CNN input is
    scale-consistent across scans acquired with different spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        max(1, round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing_mm)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing_mm)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(int(DEFAULT_HU_WINDOW[0]))  # pad with "air"
    return resampler.Execute(image)


def apply_hu_window(
    array: np.ndarray, window: tuple[float, float] = DEFAULT_HU_WINDOW
) -> np.ndarray:
    """Clip to a HU window then scale to [0, 1]."""
    low, high = window
    clipped = np.clip(array, low, high)
    return (clipped - low) / (high - low)


def preprocess_ct(
    path: str | Path,
    target_spacing_mm: tuple[float, float, float] = DEFAULT_SPACING_MM,
    hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
) -> tuple[np.ndarray, CTMetadata]:
    """Full pipeline: load -> validate -> HU convert -> resample ->
    HU-window -> normalize.

    Returns a (D, H, W) float32 array in [0, 1] plus the original volume
    metadata (recorded before resampling, for traceability/audit).
    """
    image = load_ct_volume(path)
    metadata = extract_metadata(image)

    image = hu_convert(image)
    image = resample_isotropic(image, target_spacing_mm)

    array = sitk.GetArrayFromImage(image).astype(np.float32)  # (D, H, W)
    array = apply_hu_window(array, hu_window)

    return array, metadata
