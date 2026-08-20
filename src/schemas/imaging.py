"""Schemas for patient/study metadata and vision-model findings.

Field names mirror the Response Contract in PROJECT_SPEC.md Section 31.
These are data shapes only — no inference logic lives here (that starts
Day 3/6). Patient fields are for synthetic/demo data only, never real PHI.
"""
from typing import Literal

from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    patient_id: str
    name: str
    age: int = Field(ge=0, le=130)
    sex: Literal["Male", "Female", "Other"]


class StudyInfo(BaseModel):
    modality: Literal["xray", "ct"]
    body_region: str
    study_date: str


class Finding(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)


class VisionResult(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    heatmap_available: bool = False
    localization: str | None = None
    # Day 11 fix: `heatmap_available` alone gave the frontend no way to
    # actually fetch the Grad-CAM image -- Day 8 signaled the flag but
    # never exposed the image itself. Base64-inlined (same pattern as
    # CTPreviewResponse) rather than a separate static-file endpoint, so
    # there's no server-side file cleanup/URL-lifetime concern.
    heatmap_base64: str | None = None
    # Day 11.1: CT has no Grad-CAM equivalent (heatmap_base64 stays None
    # for every CT candidate -- see inference.py's CT scope note), so CT
    # reports had zero visual content, only text coordinates. This is NOT
    # a saliency map -- just the axial slice nearest the analyzed
    # coordinate with that location marked, giving CT its own "original
    # image where appropriate" (PROJECT_SPEC.md Section 26). Always None
    # for X-ray (heatmap_base64 already covers it there).
    location_preview_base64: str | None = None
    # X-ray only: the resized-to-224x224 image the model actually saw,
    # before the Grad-CAM heatmap was blended in -- same exact dimensions
    # as `heatmap_base64` (both come from the same resize step), so a UI
    # can show them side by side with no size mismatch and no cropping.
    # Real bug this fixes: showing the raw uploaded file (its own aspect
    # ratio) next to the always-square Grad-CAM overlay either looked
    # mismatched in size (object-contain, letterboxed) or cropped real
    # content like the "R" laterality marker (object-cover, forced fit).
    # Always None for CT.
    resized_original_base64: str | None = None


class CTPreviewResponse(BaseModel):
    """Day 11: a displayable CT slice + the metadata needed to convert a
    click on it back into the world-mm coordinate `/imaging/analyze`'s
    CT path expects. See `src/vision/ct_preview.py`."""

    slice_index: int
    num_slices: int
    width: int
    height: int
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    image_base64: str
