"""Tests for Day 11 (Day 11.1: multi-candidate) PDF report generation
(src/services/report_service.py).

Uses an isolated temp DB (same pattern as tests/test_storage.py) and a
temp output directory (monkeypatched `REPORTS_DIR`) -- never touches the
real `storage/app.db` or `reports/generated/`. Real PDF content is
verified with `pypdf`, not just "a file exists" -- see
PROGRESS_LOG.md Day 11 for the real bug this caught (a probability like
0.9999 displaying as a false "100.00%").
"""
import base64
import io

import pytest
import pypdf
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.services.report_service as report_service
from src.schemas.api import AnalysisResponse, ReportCandidateInput, SafetyInfo
from src.schemas.imaging import Finding, PatientInfo, StudyInfo, VisionResult
from src.schemas.llm import CopilotResponse
from storage.models import Base


@pytest.fixture
def isolated_report_env(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path/'test.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr("storage.repositories.get_session", TestSessionLocal)
    monkeypatch.setattr(report_service, "REPORTS_DIR", tmp_path / "generated")
    yield


def _analysis(
    analysis_id: str = "analysis-1",
    label: str = "pneumonia",
    probability: float = 0.87,
    heatmap_base64: str | None = None,
    localization: str | None = None,
    modality: str = "xray",
    location_preview_base64: str | None = None,
) -> AnalysisResponse:
    return AnalysisResponse(
        analysis_id=analysis_id,
        study_id="study-1",
        patient=PatientInfo(patient_id="PT-1", name="Test Patient", age=45, sex="Male"),
        study=StudyInfo(modality=modality, body_region="chest", study_date="2026-08-15"),
        vision=VisionResult(
            findings=[Finding(label=label, probability=probability)],
            heatmap_available=heatmap_base64 is not None,
            localization=localization,
            heatmap_base64=heatmap_base64,
            location_preview_base64=location_preview_base64,
        ),
        llm=None,
        safety=SafetyInfo(),
    )


def _copilot(
    summary: str = "The model flagged pneumonia.", findings: list[str] | None = None
) -> CopilotResponse:
    return CopilotResponse(
        summary=summary,
        findings=findings if findings is not None else ["pneumonia"],
        limitations=["Not a diagnosis."],
        requires_professional_review=True,
        provider="mock",
        model="mock-model",
        grounded=True,
        kb_sources_used=["kb-pneumonia-001"],
    )


def _candidate(**kwargs) -> ReportCandidateInput:
    return ReportCandidateInput(analysis=_analysis(**kwargs), copilot=_copilot())


def test_generate_report_pdf_writes_a_real_pdf_file(isolated_report_env):
    record = report_service.generate_report_pdf([_candidate()], "resnet50-v1")

    from pathlib import Path

    assert Path(record.pdf_path).exists()
    assert Path(record.pdf_path).stat().st_size > 0


def test_generate_report_pdf_content_includes_real_findings_and_summary(isolated_report_env):
    record = report_service.generate_report_pdf([_candidate()], "resnet50-v1")

    reader = pypdf.PdfReader(record.pdf_path)
    text = "\n".join(page.extract_text() for page in reader.pages)

    assert "PNEUMONIA" in text or "pneumonia" in text.lower()
    assert "The model flagged pneumonia." in text
    assert "resnet50-v1" in text
    assert record.id in text  # the freshly-minted report id, in the Report Metadata table


def test_generate_report_pdf_never_shows_100_percent_for_sub_one_probability(isolated_report_env):
    """Real bug found during Day 11 build: 0.9998/0.99999 rounded to
    "100.0%"/"100.00%", which reads as false certainty. Verifies the fix."""
    record = report_service.generate_report_pdf([_candidate(probability=0.99999)], "resnet50-v1")

    reader = pypdf.PdfReader(record.pdf_path)
    text = "\n".join(page.extract_text() for page in reader.pages)

    assert "100.00%" not in text
    assert "100.0%" not in text
    assert "99.99%" in text


def test_generate_report_pdf_persists_a_retrievable_record(isolated_report_env):
    from storage.repositories import get_report

    saved = report_service.generate_report_pdf([_candidate()], "resnet50-v1")

    record = get_report(saved.id)
    assert record is not None
    assert record.candidates()[0]["llm_provider"] == "mock"
    assert record.candidates()[0]["copilot_findings"] == ["pneumonia"]


def test_generate_report_pdf_rejects_empty_candidate_list(isolated_report_env):
    with pytest.raises(ValueError, match="at least one candidate"):
        report_service.generate_report_pdf([], "resnet50-v1")


def test_generate_report_pdf_normalizes_unicode_hyphens_the_llm_can_emit(isolated_report_env):
    """Real bug reported by the user live: `openai/gpt-oss-120b` (via
    Groq) emitted U+2011 (non-breaking hyphen) in compound words like
    "pattern-matching" -- ReportLab's default Helvetica font has no glyph
    for it, so it rendered as a black box in the middle of a sentence.
    A plain ASCII hyphen must survive unmodified; the Unicode variant
    must be normalized to one, not silently dropped from the text."""
    summary_with_unicode_hyphen = "Consistent with pattern‑matching of non‑nodule tissue."
    record = report_service.generate_report_pdf(
        [
            ReportCandidateInput(
                analysis=_analysis(),
                copilot=_copilot(summary=summary_with_unicode_hyphen),
            )
        ],
        "resnet50-v1",
    )

    reader = pypdf.PdfReader(record.pdf_path)
    text = "\n".join(page.extract_text() for page in reader.pages)

    assert "pattern-matching" in text
    assert "non-nodule" in text
    assert "‑" not in text


def test_generate_report_pdf_explains_missing_ct_preview_instead_of_omitting_it(isolated_report_env):
    """Real bug reported by the user live: a CT report's PDF had no
    visual section at all and no explanation when the location preview
    wasn't available -- read as "the image is missing" rather than an
    explained absence."""
    record = report_service.generate_report_pdf(
        [_candidate(modality="ct", heatmap_base64=None, location_preview_base64=None)],
        "nodule-3d-cnn-v1",
    )

    reader = pypdf.PdfReader(record.pdf_path)
    text = "\n".join(page.extract_text() for page in reader.pages)

    assert "Candidate Location Preview" in text
    assert "preview rendering failed" in text


def test_generate_report_pdf_embeds_a_real_ct_location_preview_image(isolated_report_env):
    """Day 11.1's real feature: CT has no Grad-CAM, so the marked-slice
    preview (src/vision/ct_preview.py's render_candidate_preview_png) is
    CT's own "original image where appropriate" (PROJECT_SPEC.md Section
    26) -- must actually embed, not just print explanatory text."""
    img = Image.new("L", (48, 48), color=180)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    preview_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    record = report_service.generate_report_pdf(
        [_candidate(modality="ct", heatmap_base64=None, location_preview_base64=preview_b64)],
        "nodule-3d-cnn-v1",
    )

    reader = pypdf.PdfReader(record.pdf_path)
    assert any(len(page.images) > 0 for page in reader.pages)
    # Collapse whitespace -- pypdf's extraction inserts a literal newline
    # at wrapped line breaks inside a paragraph, which would otherwise
    # split a phrase that reads fine in the actual rendered PDF.
    text = " ".join(page.extract_text() for page in reader.pages).replace("\n", " ")
    assert "Candidate Location Preview" in text
    assert "not a saliency map" in text.lower()


def test_generate_report_pdf_with_multiple_candidates(isolated_report_env):
    """Day 11.1: a single CT scan with several analyzed nodule locations
    -- the real-world case this feature exists for."""
    candidates = [
        ReportCandidateInput(
            analysis=_analysis(
                analysis_id="analysis-1", label="nodule", probability=0.91, localization="RUL"
            ),
            copilot=_copilot(summary="First candidate flagged as a nodule.", findings=["nodule"]),
        ),
        ReportCandidateInput(
            analysis=_analysis(
                analysis_id="analysis-2", label="non-nodule", probability=0.88, localization="LLL"
            ),
            copilot=_copilot(summary="Second candidate not flagged as a nodule.", findings=["non-nodule"]),
        ),
    ]

    record = report_service.generate_report_pdf(candidates, "nodule-3d-cnn-v1")

    reader = pypdf.PdfReader(record.pdf_path)
    text = "\n".join(page.extract_text() for page in reader.pages)

    assert "Candidate Location 1 of 2" in text
    assert "Candidate Location 2 of 2" in text
    assert "First candidate flagged as a nodule." in text
    assert "Second candidate not flagged as a nodule." in text
    assert "RUL" in text
    assert "LLL" in text

    stored = record.candidates()
    assert len(stored) == 2
    assert stored[0]["copilot_findings"] == ["nodule"]
    assert stored[1]["copilot_findings"] == ["non-nodule"]
