"""FastAPI application entrypoint — Day 1 foundation. Imaging analysis
added Day 8; CT slice preview, Copilot (report/ask), and PDF report
generation + history (Section 30 of PROJECT_SPEC.md) added Day 11.
"""
import asyncio
import base64
import io
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import ValidationError

from src.config import get_settings
from src.llm.base import LLMConfigurationError
from src.llm.gateway import get_llm_providers
from src.preprocessing.preprocess_3d import InvalidCTError
from src.safety.input_guard import InputValidationError
from src.schemas.api import (
    AnalysisResponse,
    ErrorResponse,
    HealthResponse,
    ReportDetail,
    ReportGenerateRequest,
    ReportSummary,
    SystemInfoResponse,
)
from src.schemas.imaging import CTPreviewResponse, PatientInfo
from src.schemas.llm import CopilotAskRequest, CopilotRequest, CopilotResponse
from src.services.copilot_service import CopilotError, answer_question, generate_report
from src.services.imaging_service import AnalysisError, analyze_ct_study, analyze_xray_study
from src.services.report_service import generate_report_pdf, to_detail, to_summary
from src.utils.logging import configure_logging, get_logger
from src.vision.ct_preview import InvalidSliceError, extract_display_slice
from src.vision.inference import ModelUnavailableError
from storage.database import init_db
from storage.repositories import delete_report, get_report, list_reports

configure_logging()
logger = get_logger(__name__)
settings = get_settings()
init_db()  # Day 11: creates storage/app.db's tables if they don't exist yet

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "Medical Imaging AI Copilot — portfolio/research prototype. "
        "AI-generated output only; not a diagnostic device."
    ),
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail="Internal server error").model_dump(),
    )


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(app_name=settings.app_name, app_env=settings.app_env)


@app.get(
    f"{settings.api_v1_prefix}/system/info",
    response_model=SystemInfoResponse,
    tags=["system"],
)
async def system_info() -> SystemInfoResponse:
    """Day 12: read-only system configuration for the Settings page.
    Reflects real `get_settings()` values and real filesystem checks for
    the trained checkpoints -- never a hardcoded/fake status."""
    model_2d_path = Path(settings.model_2d_checkpoint_path)
    model_3d_path = Path(settings.model_3d_checkpoint_path)
    active_model = {
        "groq": settings.groq_model,
        "ollama": settings.ollama_model,
        "claude": settings.claude_model,
    }.get(settings.llm_provider)
    fallback_model = (
        {
            "groq": settings.groq_model,
            "ollama": settings.ollama_model,
            "claude": settings.claude_model,
        }.get(settings.llm_fallback_provider)
        if settings.llm_fallback_provider
        else None
    )
    return SystemInfoResponse(
        app_name=settings.app_name,
        app_env=settings.app_env,
        api_version=app.version,
        llm_provider=settings.llm_provider,
        llm_fallback_provider=settings.llm_fallback_provider,
        llm_model=active_model,
        llm_fallback_model=fallback_model,
        model_2d_available=model_2d_path.exists(),
        model_2d_version="resnet50-2d-v1",
        model_3d_available=model_3d_path.exists(),
        model_3d_version="nodule-3d-cnn-v1",
    )


@app.post(
    f"{settings.api_v1_prefix}/imaging/analyze",
    response_model=AnalysisResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["imaging"],
)
async def analyze_imaging(
    modality: str = Form(..., description="'xray' or 'ct'"),
    file: UploadFile = File(..., description="X-ray image, or CT .mhd header file"),
    raw_file: UploadFile | None = File(
        None, description="CT .raw data file — required when modality='ct'"
    ),
    coord_x: float | None = Form(None, description="CT candidate X coord (mm) — required for ct"),
    coord_y: float | None = Form(None, description="CT candidate Y coord (mm) — required for ct"),
    coord_z: float | None = Form(None, description="CT candidate Z coord (mm) — required for ct"),
    patient_id: str = Form(...),
    patient_name: str = Form(...),
    patient_age: int = Form(...),
    patient_sex: str = Form(..., description="'Male', 'Female', or 'Other'"),
    body_region: str = Form("chest"),
    study_date: str | None = Form(None),
    study_id: str | None = Form(None),
) -> AnalysisResponse:
    """Day 8 unified vision inference endpoint. X-ray: upload the image,
    the whole image is analyzed. CT: upload the .mhd + .raw pair AND a
    candidate coordinate — the 3D model classifies that one location, it
    does not scan a whole volume on its own (see
    `src/vision/inference.py`'s module docstring for why).

    Never returns a fabricated prediction: an unavailable checkpoint is a
    503, not a made-up result.
    """
    if modality not in ("xray", "ct"):
        raise HTTPException(422, detail=f"Unsupported modality: {modality!r}. Must be 'xray' or 'ct'.")

    try:
        patient = PatientInfo(
            patient_id=patient_id, name=patient_name, age=patient_age, sex=patient_sex
        )
    except ValidationError as exc:
        raise HTTPException(422, detail=str(exc)) from exc

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        if modality == "xray":
            image_path = tmp_path / (file.filename or "upload.jpg")
            image_path.write_bytes(await file.read())
            try:
                return analyze_xray_study(image_path, patient, body_region, study_date, study_id)
            except AnalysisError as exc:
                raise HTTPException(422, detail=str(exc)) from exc
            except ModelUnavailableError as exc:
                raise HTTPException(503, detail=str(exc)) from exc

        # modality == "ct"
        if raw_file is None:
            raise HTTPException(422, detail="CT analysis requires 'raw_file' (.raw data file).")
        if coord_x is None or coord_y is None or coord_z is None:
            raise HTTPException(422, detail="CT analysis requires coord_x, coord_y, coord_z.")

        # ElementDataFile inside the .mhd header references the .raw file
        # by its exact filename — preserve both uploads' original names,
        # in the same directory, or SimpleITK can't resolve the pair.
        mhd_path = tmp_path / (file.filename or "upload.mhd")
        mhd_path.write_bytes(await file.read())
        raw_path = tmp_path / (raw_file.filename or "upload.raw")
        raw_path.write_bytes(await raw_file.read())

        try:
            return analyze_ct_study(
                mhd_path,
                (coord_x, coord_y, coord_z),
                patient,
                body_region,
                study_date,
                study_id,
            )
        except AnalysisError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        except ModelUnavailableError as exc:
            raise HTTPException(503, detail=str(exc)) from exc


@app.post(
    f"{settings.api_v1_prefix}/imaging/ct-preview",
    response_model=CTPreviewResponse,
    responses={422: {"model": ErrorResponse}},
    tags=["imaging"],
)
async def ct_preview(
    file: UploadFile = File(..., description="CT .mhd header file"),
    raw_file: UploadFile = File(..., description="CT .raw data file"),
    slice_index: int | None = Form(None, description="Defaults to the middle slice"),
) -> CTPreviewResponse:
    """Day 11: renders one axial CT slice as a displayable PNG, plus the
    origin/spacing metadata needed to convert a click on it into the
    world-mm coordinate `/imaging/analyze`'s CT path expects (see
    `src/vision/ct_preview.py`'s module docstring). Does not run the
    3D model — this is purely for the frontend's click-to-pick UI.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Same filename-preservation requirement as /imaging/analyze's CT
        # path -- the .mhd header references the .raw file by exact name.
        mhd_path = tmp_path / (file.filename or "upload.mhd")
        mhd_path.write_bytes(await file.read())
        raw_path = tmp_path / (raw_file.filename or "upload.raw")
        raw_path.write_bytes(await raw_file.read())

        try:
            result = extract_display_slice(mhd_path, slice_index=slice_index)
        except InvalidCTError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        except InvalidSliceError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

    buffer = io.BytesIO()
    Image.fromarray(result.image_array, mode="L").save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return CTPreviewResponse(
        slice_index=result.slice_index,
        num_slices=result.num_slices,
        width=result.width,
        height=result.height,
        origin=result.origin,
        spacing=result.spacing,
        image_base64=image_base64,
    )


def _to_copilot_response(result, kb_sources_used: list[str] | None = None) -> CopilotResponse:
    return CopilotResponse(
        summary=result.report.summary,
        findings=result.report.findings,
        limitations=result.report.limitations,
        requires_professional_review=result.report.requires_professional_review,
        provider=result.provider,
        model=result.model,
        grounded=result.grounded,
        kb_sources_used=result.kb_sources_used,
    )


@app.post(
    f"{settings.api_v1_prefix}/copilot/report",
    response_model=CopilotResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["copilot"],
)
async def copilot_report(request: CopilotRequest) -> CopilotResponse:
    """Day 11: grounded preliminary-report generation (Section 8),
    wiring Day 10's `src/services/copilot_service.py` to real HTTP.
    Takes findings from a prior `/imaging/analyze` call -- the LLM never
    sees the image, only these structured findings. Falls back to
    LLM_FALLBACK_PROVIDER (if configured) when the primary provider
    can't produce a valid response.
    """
    try:
        providers = get_llm_providers()
    except LLMConfigurationError as exc:
        raise HTTPException(503, detail=str(exc)) from exc

    try:
        result = await generate_report(providers, request.findings, request.localization, request.modality)
    except CopilotError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    return _to_copilot_response(result)


@app.post(
    f"{settings.api_v1_prefix}/copilot/ask",
    response_model=CopilotResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["copilot"],
)
async def copilot_ask(request: CopilotAskRequest) -> CopilotResponse:
    """Day 11: Copilot Q&A about the current analysis (Section 9),
    wiring Day 10's `answer_question()` to real HTTP. Falls back to
    LLM_FALLBACK_PROVIDER (if configured) when the primary provider
    can't produce a valid response."""
    try:
        providers = get_llm_providers()
    except LLMConfigurationError as exc:
        raise HTTPException(503, detail=str(exc)) from exc

    try:
        result = await answer_question(
            providers,
            request.question,
            request.findings,
            request.localization,
            request.modality,
            request.conversation_history,
        )
    except InputValidationError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    except CopilotError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    return _to_copilot_response(result)


@app.post(
    f"{settings.api_v1_prefix}/report/generate",
    response_model=ReportSummary,
    responses={422: {"model": ErrorResponse}},
    tags=["report"],
)
async def report_generate(request: ReportGenerateRequest) -> ReportSummary:
    """Day 11 (Day 11.1: multi-candidate): renders the professional PDF
    (Section 26) from one or more prior /imaging/analyze + /copilot/*
    response pairs, and persists it so it shows up in Reports History.
    X-ray callers pass exactly one candidate; CT callers may pass several
    -- one per candidate location analyzed on the same scan. PDF
    generation is sync/CPU-bound (reportlab) -- dispatched via
    `asyncio.to_thread` so it doesn't block the event loop."""
    try:
        record = await asyncio.to_thread(
            generate_report_pdf, request.candidates, request.vision_model_version
        )
    except Exception as exc:  # noqa: BLE001 -- surfaced as a clear 422, not a bare 500
        raise HTTPException(422, detail=f"Could not generate report: {exc}") from exc
    return to_summary(record)


@app.get(
    f"{settings.api_v1_prefix}/reports",
    response_model=list[ReportSummary],
    tags=["report"],
)
async def reports_list(limit: int = 100) -> list[ReportSummary]:
    """Day 11: Reports History list, backed by the real SQLite store."""
    records = await asyncio.to_thread(list_reports, limit)
    return [to_summary(r) for r in records]


@app.get(
    f"{settings.api_v1_prefix}/reports/{{report_id}}",
    response_model=ReportDetail,
    responses={404: {"model": ErrorResponse}},
    tags=["report"],
)
async def report_detail(report_id: str) -> ReportDetail:
    record = await asyncio.to_thread(get_report, report_id)
    if record is None:
        raise HTTPException(404, detail=f"No report with id {report_id!r}.")
    return to_detail(record)


@app.get(
    f"{settings.api_v1_prefix}/reports/{{report_id}}/pdf",
    responses={404: {"model": ErrorResponse}},
    tags=["report"],
)
async def report_pdf(report_id: str) -> FileResponse:
    record = await asyncio.to_thread(get_report, report_id)
    if record is None or not Path(record.pdf_path).exists():
        raise HTTPException(404, detail=f"No report PDF for id {report_id!r}.")
    return FileResponse(
        record.pdf_path,
        media_type="application/pdf",
        filename=f"medical-imaging-report-{report_id}.pdf",
    )


@app.delete(
    f"{settings.api_v1_prefix}/reports/{{report_id}}",
    status_code=204,
    responses={404: {"model": ErrorResponse}},
    tags=["report"],
)
async def report_delete(report_id: str) -> Response:
    """Day 12: permanently deletes a report -- the DB record and its
    generated PDF file. The DB delete is the source of truth (404 if it
    didn't exist); the PDF file removal is best-effort after that (a
    missing/already-gone file just gets logged, never turned into a
    500 -- the report is still correctly gone either way)."""
    pdf_path = await asyncio.to_thread(delete_report, report_id)
    if pdf_path is None:
        raise HTTPException(404, detail=f"No report with id {report_id!r}.")
    try:
        Path(pdf_path).unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove PDF file %s for deleted report %s", pdf_path, report_id)
    return Response(status_code=204)
