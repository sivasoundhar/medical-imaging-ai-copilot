"""CT slice preview + pixel-to-world coordinate conversion (Day 11).

Supports the frontend's click-to-pick coordinate UI: the user needs to
*see* a CT slice before they can click a point on it, and the backend
needs to convert that click back into the world-mm coordinate
`/api/v1/imaging/analyze`'s CT path expects.

Reuses the exact same `load_ct_volume` -> `resample_isotropic` pipeline
`src/vision/dataset_3d.py`'s `extract_candidate_patch` uses, so a
coordinate computed from a slice this module renders lands in the same
world-mm space the model was trained on -- not a second, potentially
inconsistent coordinate system.

Coordinate math here assumes axis-aligned direction cosines (no gantry
tilt / rotation) -- true for LUNA16's real volumes (verified in Day
6/7's real-data checks), but a known simplification if a future dataset
has rotated volumes. Flagged, not silently assumed correct.
"""
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw

from src.preprocessing.preprocess_3d import (
    DEFAULT_HU_WINDOW,
    DEFAULT_SPACING_MM,
    apply_hu_window,
    load_ct_volume,
    resample_isotropic,
)

# Matches the project's warn/danger red (see frontend's Tailwind palette)
# -- not a decorative color choice, reserved for flagging an actual
# location the model classified, same convention as the rest of the app.
_MARKER_COLOR = (255, 60, 60)


class InvalidSliceError(ValueError):
    """Raised for an out-of-range `slice_index`."""


@dataclass
class CTPreviewSlice:
    image_array: np.ndarray  # (H, W) uint8, ready to PNG-encode
    slice_index: int
    num_slices: int
    width: int
    height: int
    origin: tuple[float, float, float]  # resampled image's world origin (x, y, z) mm
    spacing: tuple[float, float, float]  # resampled image's spacing (x, y, z) mm -- isotropic


def extract_display_slice(
    volume_path: str | Path,
    slice_index: int | None = None,
    hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
    target_spacing_mm: tuple[float, float, float] = DEFAULT_SPACING_MM,
) -> CTPreviewSlice:
    """Loads a CT volume, resamples to isotropic spacing (same pipeline
    as model inference), and extracts one axial slice as an HU-windowed
    8-bit grayscale image. `slice_index` defaults to the middle slice.

    Raises `InvalidCTError` (from `load_ct_volume`) for a bad file, or
    `InvalidSliceError` for an out-of-range index.
    """
    image = load_ct_volume(volume_path)
    resampled = resample_isotropic(image, target_spacing_mm)
    array = sitk.GetArrayFromImage(resampled)  # (D, H, W) -- standard ITK/numpy axis order
    num_slices = array.shape[0]

    if slice_index is None:
        slice_index = num_slices // 2
    if not (0 <= slice_index < num_slices):
        raise InvalidSliceError(
            f"slice_index {slice_index} out of range for volume with {num_slices} slices."
        )

    windowed = apply_hu_window(array[slice_index], hu_window)  # (H, W) float32 in [0, 1]
    image_uint8 = (windowed * 255).astype(np.uint8)

    return CTPreviewSlice(
        image_array=image_uint8,
        slice_index=slice_index,
        num_slices=num_slices,
        width=image_uint8.shape[1],
        height=image_uint8.shape[0],
        origin=resampled.GetOrigin(),
        spacing=resampled.GetSpacing(),
    )


def pixel_to_world(
    row: int,
    col: int,
    slice_index: int,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Converts a click at (row, col) on the slice at `slice_index` back
    to world-mm (x, y, z) -- the exact inverse of the axis mapping
    `dataset_3d.py`'s `crop_patch_from_resampled` uses
    (`TransformPhysicalPointToContinuousIndex` -> `(ix, iy, iz)` ->
    numpy `(iz, iy, ix)`), so a coordinate picked from this module's
    preview lands in the same space `extract_candidate_patch` expects.

    Assumes axis-aligned direction cosines (see module docstring).
    """
    ox, oy, oz = origin
    sx, sy, sz = spacing
    world_x = ox + col * sx
    world_y = oy + row * sy
    world_z = oz + slice_index * sz
    return (world_x, world_y, world_z)


def world_to_pixel(
    world_xyz: tuple[float, float, float],
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
) -> tuple[int, int, int]:
    """The inverse of `pixel_to_world`: converts a world-mm (x, y, z)
    coordinate -- e.g. one a candidate was actually analyzed at -- back to
    (row, col, slice_index) in the resampled volume this module's
    `extract_display_slice` slices come from. Used to mark a candidate's
    real analyzed location on a report preview image (Day 11.1)."""
    wx, wy, wz = world_xyz
    ox, oy, oz = origin
    sx, sy, sz = spacing
    col = round((wx - ox) / sx)
    row = round((wy - oy) / sy)
    slice_index = round((wz - oz) / sz)
    return row, col, slice_index


def _draw_candidate_marker(image_uint8: np.ndarray, row: int, col: int, radius: int = 12) -> np.ndarray:
    """Draws a circle + crosshair ticks at (row, col) on a grayscale
    uint8 slice, returning an RGB array. Pure PIL (already a project
    dependency via `gradcam.py`) -- no new dependency added."""
    rgb = Image.fromarray(image_uint8).convert("RGB")
    draw = ImageDraw.Draw(rgb)
    draw.ellipse([col - radius, row - radius, col + radius, row + radius], outline=_MARKER_COLOR, width=2)
    tick = radius + 6
    draw.line([col - tick, row, col - radius, row], fill=_MARKER_COLOR, width=2)
    draw.line([col + radius, row, col + tick, row], fill=_MARKER_COLOR, width=2)
    draw.line([col, row - tick, col, row - radius], fill=_MARKER_COLOR, width=2)
    draw.line([col, row + radius, col, row + tick], fill=_MARKER_COLOR, width=2)
    return np.array(rgb)


def render_candidate_preview_png(
    volume_path: str | Path,
    coord_xyz: tuple[float, float, float],
    hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
    target_spacing_mm: tuple[float, float, float] = DEFAULT_SPACING_MM,
) -> bytes:
    """Renders the axial slice nearest a CT candidate's actual analyzed
    coordinate, marked with its location, as PNG bytes -- gives CT
    reports/analysis responses an actual picture of the scan
    (PROJECT_SPEC.md Section 26: "original image where appropriate"),
    since the 3D model has no Grad-CAM equivalent (see
    `inference.py`'s CT scope note; this is NOT a saliency map, just a
    "here's where we looked" marker).

    Reuses the exact same `load_ct_volume` -> `resample_isotropic`
    pipeline `extract_candidate_patch`/`extract_display_slice` use, and
    `world_to_pixel` (this module's own documented inverse of
    `pixel_to_world`), so the marker lands exactly where the model
    actually classified -- not a second, potentially inconsistent
    estimate of the same point.
    """
    image = load_ct_volume(volume_path)
    resampled = resample_isotropic(image, target_spacing_mm)
    array = sitk.GetArrayFromImage(resampled)  # (D, H, W)
    num_slices = array.shape[0]
    origin = resampled.GetOrigin()
    spacing = resampled.GetSpacing()

    row, col, slice_index = world_to_pixel(coord_xyz, origin, spacing)
    # Clamp defensively -- a candidate near a volume's edge could in
    # principle round to a slice index just outside range; this is a
    # display preview, not the model's own crop (that already succeeded
    # by the time this runs), so clamping to the nearest real slice is
    # correct here rather than raising and losing the whole preview.
    slice_index = max(0, min(slice_index, num_slices - 1))

    windowed = apply_hu_window(array[slice_index], hu_window)
    image_uint8 = (windowed * 255).astype(np.uint8)
    marked = _draw_candidate_marker(image_uint8, row, col)

    buf = io.BytesIO()
    Image.fromarray(marked).save(buf, format="PNG")
    return buf.getvalue()
