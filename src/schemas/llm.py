"""Schemas for the LLM gateway layer (wired up starting Day 9).

`LLMResult` is kept here since Day 1 only so the response contract
(schemas/api.py, Section 31) has something concrete to reference --
it's the thin API-contract envelope (`{provider, model, report: str,
grounded}`), not touched by Day 10.

`MedicalReport` (Day 10, PROJECT_SPEC.md Section 21) is the structured
shape the LLM itself is instructed to return for the Medical Copilot
(report generation + Q&A) -- validated with Pydantic, then checked for
groundedness and safety (src/safety/) before ever reaching a user.

`CopilotRequest`/`CopilotAskRequest`/`CopilotResponse` (Day 11) are the
HTTP request/response shapes for `/api/v1/copilot/*`.
"""
from pydantic import BaseModel, Field

from src.schemas.imaging import Finding


class LLMResult(BaseModel):
    provider: str
    model: str
    report: str
    grounded: bool


class MedicalReport(BaseModel):
    """PROJECT_SPEC.md Section 21's exact structured-output shape."""

    summary: str
    findings: list[str]
    limitations: list[str]
    requires_professional_review: bool


class CopilotRequest(BaseModel):
    """HTTP request shape for `/api/v1/copilot/report` -- findings come
    from a prior `/api/v1/imaging/analyze` call, never re-derived from
    an image here (the LLM never sees the image)."""

    findings: list[Finding]
    localization: str | None = None
    modality: str = "xray"


class CopilotAskRequest(CopilotRequest):
    """HTTP request shape for `/api/v1/copilot/ask`."""

    question: str
    conversation_history: list[dict] | None = None


class CopilotResponse(BaseModel):
    """HTTP response shape wrapping `MedicalReport` + provenance
    metadata (Section 49's determinism requirement: "record
    provider/model")."""

    summary: str
    findings: list[str]
    limitations: list[str]
    requires_professional_review: bool
    provider: str
    model: str
    grounded: bool
    kb_sources_used: list[str] = Field(default_factory=list)
