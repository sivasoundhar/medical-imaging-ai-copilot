"""Day 8 orchestration (PROJECT_SPEC.md Section 29's
`src/services/imaging_service.py`): ties preprocessing + `src/vision/
inference.py` together into the full `AnalysisResponse` shape (Section
31).

FastAPI-specific concerns (`UploadFile`, multipart form parsing) stay in
`src/main.py` — this module works with plain file paths so it's testable
without spinning up the app or mocking HTTP internals.
"""
import base64
import uuid
from datetime import date
from pathlib import Path

from src.config import get_settings
from src.preprocessing.preprocess_2d import InvalidXrayError
from src.preprocessing.preprocess_3d import InvalidCTError
from src.schemas.api import AnalysisResponse, SafetyInfo
from src.schemas.imaging import Finding, PatientInfo, StudyInfo, VisionResult
from src.utils.logging import get_logger
from src.vision.ct_preview import render_candidate_preview_png
from src.vision.inference import analyze_ct, analyze_xray

logger = get_logger(__name__)

__all__ = [
    "AnalysisError",
    "analyze_xray_study",
    "analyze_ct_study",
]


class AnalysisError(ValueError):
    """Wraps modality-specific validation failures (bad image/volume/
    coordinates) into one type the API layer can catch and turn into a
    422, without needing to know about each preprocessing module's own
    exception type."""


def analyze_xray_study(
    image_path: str | Path,
    patient: PatientInfo,
    body_region: str = "chest",
    study_date: str | None = None,
    study_id: str | None = None,
) -> AnalysisResponse:
    """Raises `AnalysisError` for a bad image, `ModelUnavailableError`
    (from `src.vision.inference`, uncaught here — the caller decides how
    to map it, e.g. HTTP 503) if the checkpoint is missing."""
    settings = get_settings()
    try:
        result = analyze_xray(image_path, settings.model_2d_checkpoint_path)
    except InvalidXrayError as exc:
        raise AnalysisError(str(exc)) from exc

    # Day 11 fix: `result.heatmap_path` was computed but previously
    # discarded here -- `heatmap_available: true` with no way to fetch
    # the actual image. Read it back and inline it.
    heatmap_path = Path(result.heatmap_path)
    heatmap_base64 = (
        base64.b64encode(heatmap_path.read_bytes()).decode("ascii")
        if heatmap_path.exists()
        else None
    )
    # Same treatment for the resized original (see VisionResult.
    # resized_original_base64's docstring for why this exists) -- best
    # effort, same as the heatmap: a missing file degrades to None rather
    # than failing an otherwise-successful classification.
    resized_original_path = Path(result.resized_original_path)
    resized_original_base64 = (
        base64.b64encode(resized_original_path.read_bytes()).decode("ascii")
        if resized_original_path.exists()
        else None
    )

    vision = VisionResult(
        findings=[Finding(label=result.prediction, probability=result.probability)],
        heatmap_available=heatmap_base64 is not None,
        localization=None,
        heatmap_base64=heatmap_base64,
        resized_original_base64=resized_original_base64,
    )
    return _build_response(patient, "xray", body_region, study_date, study_id, vision)


def analyze_ct_study(
    volume_path: str | Path,
    coord_xyz: tuple[float, float, float],
    patient: PatientInfo,
    body_region: str = "chest",
    study_date: str | None = None,
    study_id: str | None = None,
) -> AnalysisResponse:
    """Raises `AnalysisError` for a bad volume, `ModelUnavailableError`
    if the checkpoint is missing. `coord_xyz` is required — see
    `src/vision/inference.py`'s module docstring for why (the 3D model
    classifies a candidate location, it doesn't scan a whole volume)."""
    settings = get_settings()
    try:
        result = analyze_ct(volume_path, coord_xyz, settings.model_3d_checkpoint_path)
    except InvalidCTError as exc:
        raise AnalysisError(str(exc)) from exc

    # Day 11.1: no 3D Grad-CAM equivalent exists, so give CT its own
    # visual instead -- the analyzed slice with the candidate location
    # marked (not a saliency map). Best-effort: a rendering failure here
    # shouldn't take down an otherwise-successful classification, so it
    # degrades to no preview rather than raising.
    try:
        location_preview_base64 = base64.b64encode(
            render_candidate_preview_png(volume_path, coord_xyz)
        ).decode("ascii")
    except Exception:
        logger.warning("CT candidate preview rendering failed; continuing without it.", exc_info=True)
        location_preview_base64 = None

    vision = VisionResult(
        findings=[Finding(label=result.prediction, probability=result.probability)],
        heatmap_available=False,  # no 3D Grad-CAM built (Day 5 covered 2D only)
        localization=(
            f"({coord_xyz[0]:.1f}, {coord_xyz[1]:.1f}, {coord_xyz[2]:.1f}) mm"
        ),
        location_preview_base64=location_preview_base64,
    )
    return _build_response(patient, "ct", body_region, study_date, study_id, vision)


def _generate_study_id() -> str:
    """Fallback when no study_id was supplied. A bare uuid4() ("real,
    unique) reads as fake to a human -- an 8-char hex suffix in an
    accession-number-like format is just as unique for this project's
    purposes but looks like the real thing (matches the frontend's own
    STU-YYYY-##### convention -- see PatientForm.tsx)."""
    return f"STU-{uuid.uuid4().hex[:8].upper()}"


def _build_response(
    patient: PatientInfo,
    modality: str,
    body_region: str,
    study_date: str | None,
    study_id: str | None,
    vision: VisionResult,
) -> AnalysisResponse:
    return AnalysisResponse(
        analysis_id=str(uuid.uuid4()),
        study_id=study_id or _generate_study_id(),
        patient=patient,
        study=StudyInfo(
            modality=modality,
            body_region=body_region,
            study_date=study_date or date.today().isoformat(),
        ),
        vision=vision,
        llm=None,  # Day 9 wires this up
        safety=SafetyInfo(),
    )
