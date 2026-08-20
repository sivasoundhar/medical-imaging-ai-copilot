"""Top-level API request/response schemas."""
from pydantic import BaseModel, Field

from src.schemas.imaging import Finding, PatientInfo, StudyInfo, VisionResult
from src.schemas.llm import CopilotResponse, LLMResult


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    app_env: str


class SafetyInfo(BaseModel):
    requires_professional_review: bool = True


class AnalysisResponse(BaseModel):
    """Full response contract — see PROJECT_SPEC.md Section 31.

    Not wired to real inference yet (that starts Day 3/6/8); defined now
    so later days implement against a fixed, agreed shape.
    """

    analysis_id: str
    study_id: str
    patient: PatientInfo
    study: StudyInfo
    vision: VisionResult
    llm: LLMResult | None = None
    safety: SafetyInfo = SafetyInfo()


class ErrorResponse(BaseModel):
    detail: str


class SystemInfoResponse(BaseModel):
    """Day 12: read-only system configuration for the Settings page. Never
    includes secrets (API keys) — only what provider/model/model-version
    is active, and whether each checkpoint file the app was configured to
    use actually exists on disk. All values are read straight from
    `get_settings()` / real filesystem checks, nothing hardcoded."""

    app_name: str
    app_env: str
    api_version: str
    llm_provider: str
    llm_fallback_provider: str | None
    llm_model: str | None
    llm_fallback_model: str | None
    model_2d_available: bool
    model_2d_version: str
    model_3d_available: bool
    model_3d_version: str


class ReportCandidateInput(BaseModel):
    """One analyzed location's worth of input to a report. A prior
    /imaging/analyze response paired with a prior /copilot/report or
    /copilot/ask response for that same analysis -- the report generator
    never re-runs inference itself, only formats what already happened.

    X-ray reports always have exactly one candidate (the whole image is
    one analysis). CT reports can have several -- one per candidate
    coordinate the user picked and analyzed on the same scan (Section 26
    real-world requirement: a single CT read can cover multiple nodules)."""

    analysis: AnalysisResponse
    copilot: CopilotResponse


class ReportGenerateRequest(BaseModel):
    """Day 11/11.1: request body for POST /api/v1/report/generate."""

    candidates: list[ReportCandidateInput] = Field(min_length=1)
    vision_model_version: str = "unknown"


class ReportSummary(BaseModel):
    """One row in Reports History."""

    report_id: str
    study_id: str
    patient_name: str
    modality: str
    study_date: str
    primary_finding: str | None
    location_count: int = 1
    created_at: str


class ReportCandidateDetail(BaseModel):
    """One candidate location's findings + grounded explanation, as
    persisted and redisplayed -- mirrors `ReportCandidateInput` but
    flattened to what's actually needed for redisplay (no raw image
    bytes stored; the PDF embeds the Grad-CAM at generation time)."""

    localization: str | None
    vision_findings: list[Finding]
    copilot_summary: str
    copilot_findings: list[str]
    copilot_limitations: list[str]
    requires_professional_review: bool
    llm_provider: str
    llm_model: str


class ReportDetail(ReportSummary):
    """Full report metadata (not the PDF bytes themselves — see
    GET /api/v1/reports/{id}/pdf for that)."""

    patient_id: str
    patient_age: int
    patient_sex: str
    body_region: str
    vision_model_version: str
    candidates: list[ReportCandidateDetail]
