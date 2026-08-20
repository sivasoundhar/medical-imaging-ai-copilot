"""Day 11 professional PDF report generation (PROJECT_SPEC.md Section
26's required content: patient/study metadata, original image where
appropriate, prediction, model probability, Grad-CAM/localization,
AI-generated preliminary impression, limitations, professional-review
disclaimer, model version, LLM provider/model, analysis ID, report
timestamp).

Ties together one or more candidates -- each an `AnalysisResponse` (Day
8's vision result) paired with a `CopilotResponse` (Day 11's grounded
Copilot explanation) -- into one PDF, then persists a record via
`storage/repositories.py` so it shows up in Reports History. More than
one candidate covers the real-world case of a single CT scan with
several analyzed nodule locations (Day 11.1); X-ray reports always pass
exactly one. Uses `reportlab` -- no HTML/CSS templating layer, kept
simple and dependency-light for a portfolio project.
"""
import base64
import html
import io
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.schemas.api import ReportCandidateDetail, ReportCandidateInput, ReportDetail, ReportSummary
from src.schemas.imaging import Finding
from storage.repositories import NewReport, ReportRecord, save_report

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "generated"

_STYLES = getSampleStyleSheet()
_TITLE_STYLE = ParagraphStyle(
    "ReportTitle", parent=_STYLES["Title"], textColor=colors.HexColor("#101820")
)
_DISCLAIMER_STYLE = ParagraphStyle(
    "Disclaimer",
    parent=_STYLES["BodyText"],
    textColor=colors.HexColor("#9A5B0A"),
    backColor=colors.HexColor("#FBF0DD"),
    borderColor=colors.HexColor("#E7C384"),
    borderWidth=1,
    borderPadding=8,
)
_SECTION_STYLE = ParagraphStyle(
    "Section", parent=_STYLES["Heading2"], textColor=colors.HexColor("#0E7C86")
)


# Punctuation variants a model can emit that ReportLab's default font
# (Helvetica, WinAnsi/cp1252-ish encoding -- essentially Latin-1) has no
# glyph for, rendered as a "missing glyph" black box otherwise. Real bug
# hit live: `openai/gpt-oss-120b` (via Groq) emitted U+2011 (non-breaking
# hyphen) in compound words like "pattern-matching" on one call, showing
# as a black box between words in the PDF, while a plain ASCII "-" from
# the same model on a different call rendered fine.
_UNSUPPORTED_PUNCTUATION = {
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    "‒": "-",  # figure dash
    "–": "-",  # en dash
    "—": "-",  # em dash
    "―": "-",  # horizontal bar
    "−": "-",  # minus sign
    " ": " ",  # non-breaking space
    "​": "",  # zero-width space
}


def _sanitize_glyphs(text: str) -> str:
    """Normalizes free text -- LLM output, patient-entered fields -- so it
    never renders a black "missing glyph" box: folds punctuation variants
    the default font can't display (see `_UNSUPPORTED_PUNCTUATION` above),
    then defensively maps anything still outside the font's supported
    range to a plain '?' -- so a future model emitting some other exotic
    character degrades to a readable substitution instead of the same bug
    recurring. Safe for both plain `Table` cells and `Paragraph` text."""
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _UNSUPPORTED_PUNCTUATION.items():
        text = text.replace(bad, good)
    return text.encode("cp1252", errors="replace").decode("cp1252")


def _pdf_text(text: str) -> str:
    """`_sanitize_glyphs` plus XML-escaping, for text going into a
    `Paragraph` specifically -- `Paragraph` interprets a small HTML-like
    markup subset and would otherwise raise or mis-render on a raw '&',
    '<', or '>' coming from LLM text. Do NOT use this for plain `Table`
    cell strings -- those aren't markup-parsed, so escaping would show a
    literal "&amp;" instead of "&"."""
    return html.escape(_sanitize_glyphs(text))


def _format_probability(probability: float) -> str:
    """Formats a [0, 1] probability as a percentage string that never
    displays "100.00%" for a value strictly less than 1.0 -- see the
    call site's comment for why that distinction matters here."""
    pct = probability * 100
    if probability < 1.0 and pct >= 99.995:
        pct = 99.99
    return f"{pct:.2f}%"


def generate_report_pdf(
    candidates: list[ReportCandidateInput],
    vision_model_version: str,
) -> ReportRecord:
    """Renders the PDF to `reports/generated/{report_id}.pdf` and
    persists a `ReportRecord`. `candidates` is one or more analyzed
    locations on the same patient/study/scan (Day 11.1) -- X-ray callers
    always pass exactly one; CT callers may pass several, one per
    candidate coordinate analyzed. Returns the saved record (callers use
    its `.id` / `.pdf_path`)."""
    if not candidates:
        raise ValueError("generate_report_pdf requires at least one candidate.")
    report_id = new_report_id()  # a multi-candidate report isn't 1:1 with any single analysis_id
    primary = candidates[0].analysis  # patient/study metadata is identical across candidates

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = REPORTS_DIR / f"{report_id}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    story = []

    story.append(Paragraph("Medical Imaging AI Copilot", _TITLE_STYLE))
    story.append(
        Paragraph("AI-Generated Preliminary Report — Research/Portfolio Prototype", _STYLES["Normal"])
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "<b>This is AI-generated output, not a clinical diagnosis.</b> Model "
            "probability is not clinical certainty. This system is a research/"
            "portfolio prototype and is not a substitute for a qualified "
            "healthcare professional's review.",
            _DISCLAIMER_STYLE,
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Patient &amp; Study Information", _SECTION_STYLE))
    info_table = Table(
        [
            ["Patient ID", _sanitize_glyphs(primary.patient.patient_id), "Study ID", primary.study_id],
            ["Name", _sanitize_glyphs(primary.patient.name), "Study Date", primary.study.study_date],
            [
                "Age / Sex",
                f"{primary.patient.age} / {primary.patient.sex}",
                "Modality",
                primary.study.modality.upper(),
            ],
            [
                "Body Region",
                _sanitize_glyphs(primary.study.body_region),
                "Locations Analyzed",
                str(len(candidates)),
            ],
        ],
        colWidths=[1.1 * inch, 2.1 * inch, 1.1 * inch, 2.1 * inch],
    )
    info_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#53616D")),
                ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#53616D")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DBE3E9")),
            ]
        )
    )
    story.append(info_table)
    story.append(Spacer(1, 0.2 * inch))

    multi = len(candidates) > 1
    for i, candidate in enumerate(candidates):
        analysis, copilot = candidate.analysis, candidate.copilot
        heading = f"Candidate Location {i + 1} of {len(candidates)}" if multi else "AI Vision Model Findings"
        story.append(Paragraph(heading, _SECTION_STYLE))

        # More decimals AND an explicit clamp -- a 0.9998 probability rounds
        # to "100.0%" at 1 decimal, and real observed values from this
        # project go as high as 0.99999 (Day 5 log), which still rounds to
        # "100.00%" at 2 decimals. This project never claims 100% confidence
        # (Section 3 rule 3: keep model confidence separate from clinical
        # certainty), so display precision has to actually enforce that, not
        # just word around it -- clamp anything under 1.0 to display as at
        # most 99.99%.
        findings_rows = [["Label", "Model Probability"]] + [
            [f.label, f"{_format_probability(f.probability)}"] for f in analysis.vision.findings
        ]
        if len(findings_rows) == 1:
            findings_rows.append(["(none flagged)", "—"])
        findings_table = Table(findings_rows, colWidths=[3.2 * inch, 1.5 * inch])
        findings_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F5")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DBE3E9")),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(findings_table)
        if analysis.vision.localization:
            story.append(Spacer(1, 0.08 * inch))
            story.append(
                Paragraph(f"<b>Localization:</b> {analysis.vision.localization}", _STYLES["Normal"])
            )
        story.append(Spacer(1, 0.15 * inch))

        if analysis.vision.heatmap_base64:
            story.append(Paragraph("Grad-CAM Explainability", _STYLES["Heading3"]))
            story.append(
                Paragraph(
                    "Highlights which image regions most influenced the model's "
                    "prediction. It does not prove a disease is present and is not a diagnosis.",
                    _STYLES["Normal"],
                )
            )
            story.append(Spacer(1, 0.08 * inch))
            image_bytes = base64.b64decode(analysis.vision.heatmap_base64)
            story.append(RLImage(io.BytesIO(image_bytes), width=2.5 * inch, height=2.5 * inch))
            story.append(Spacer(1, 0.15 * inch))
        elif analysis.study.modality == "ct":
            # Day 11.1: CT has no Grad-CAM equivalent (see inference.py's CT
            # scope note), so give it its own visual instead of silently
            # omitting the whole section -- the marked-slice preview
            # (src/vision/ct_preview.py's render_candidate_preview_png)
            # rendered at analysis time. Not a saliency map; say so, so the
            # reader doesn't mistake the marker for one.
            story.append(Paragraph("Candidate Location Preview", _STYLES["Heading3"]))
            if analysis.vision.location_preview_base64:
                story.append(
                    Paragraph(
                        "The analyzed CT slice nearest the candidate coordinate, with the "
                        "location marked. This is not a saliency map — Grad-CAM visualization "
                        "has only been implemented for the 2D X-ray model.",
                        _STYLES["Normal"],
                    )
                )
                story.append(Spacer(1, 0.08 * inch))
                image_bytes = base64.b64decode(analysis.vision.location_preview_base64)
                story.append(RLImage(io.BytesIO(image_bytes), width=2.5 * inch, height=2.5 * inch))
            else:
                # Best-effort rendering (imaging_service.py) failed for this
                # candidate -- say so honestly rather than pretending there
                # was never meant to be an image here.
                story.append(
                    Paragraph(
                        "Not available for this candidate — preview rendering failed. No "
                        "saliency overlay either; Grad-CAM visualization has only been "
                        "implemented for the 2D X-ray model.",
                        _STYLES["Normal"],
                    )
                )
            story.append(Spacer(1, 0.15 * inch))
        else:
            story.append(Paragraph("Grad-CAM Explainability", _STYLES["Heading3"]))
            story.append(Paragraph("No heatmap was available for this analysis.", _STYLES["Normal"]))
            story.append(Spacer(1, 0.15 * inch))

        story.append(Paragraph("AI-Generated Preliminary Impression", _STYLES["Heading3"]))
        story.append(Paragraph(_pdf_text(copilot.summary), _STYLES["BodyText"]))
        story.append(Spacer(1, 0.1 * inch))
        if copilot.limitations:
            story.append(Paragraph("<b>Limitations</b>", _STYLES["Normal"]))
            for lim in copilot.limitations:
                story.append(Paragraph(f"• {_pdf_text(lim)}", _STYLES["BodyText"]))
        story.append(
            Paragraph(
                f"<i>LLM: {copilot.provider} / {copilot.model} · Grounded: {copilot.grounded} · "
                f"Requires professional review: {copilot.requires_professional_review}</i>",
                _STYLES["Normal"],
            )
        )
        story.append(Spacer(1, 0.25 * inch))

    generated_at = datetime.now(timezone.utc)
    story.append(Paragraph("Report Metadata", _SECTION_STYLE))
    meta_table = Table(
        [
            ["Vision model version", vision_model_version],
            ["Report ID", report_id],
            ["Report generated (UTC)", generated_at.strftime("%Y-%m-%d %H:%M:%S")],
        ],
        colWidths=[2.3 * inch, 3.0 * inch],
    )
    meta_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#53616D")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DBE3E9")),
            ]
        )
    )
    story.append(meta_table)

    doc.build(story)

    record = save_report(
        NewReport(
            id=report_id,
            study_id=primary.study_id,
            patient_id=primary.patient.patient_id,
            patient_name=primary.patient.name,
            patient_age=primary.patient.age,
            patient_sex=primary.patient.sex,
            modality=primary.study.modality,
            body_region=primary.study.body_region,
            study_date=primary.study.study_date,
            candidates=[
                {
                    "localization": c.analysis.vision.localization,
                    "vision_findings": [f.model_dump() for f in c.analysis.vision.findings],
                    "heatmap_available": c.analysis.vision.heatmap_available,
                    "copilot_summary": c.copilot.summary,
                    "copilot_findings": c.copilot.findings,
                    "copilot_limitations": c.copilot.limitations,
                    "requires_professional_review": c.copilot.requires_professional_review,
                    "llm_provider": c.copilot.provider,
                    "llm_model": c.copilot.model,
                }
                for c in candidates
            ],
            vision_model_version=vision_model_version,
            pdf_path=str(pdf_path),
        )
    )
    return record


def new_report_id() -> str:
    """A raw uuid4() ("real, unique) reads as fake to a human -- same
    reasoning as `imaging_service.py`'s `_generate_study_id()`. An
    accession-number-style prefix + short hex suffix is just as unique
    for this project's purposes but looks like a real report ID."""
    return f"RPT-{uuid.uuid4().hex[:12].upper()}"


def to_summary(record: ReportRecord) -> ReportSummary:
    candidates = record.candidates()
    first_findings = candidates[0]["vision_findings"] if candidates else []
    primary_finding = first_findings[0]["label"] if first_findings else None
    return ReportSummary(
        report_id=record.id,
        study_id=record.study_id,
        patient_name=record.patient_name,
        modality=record.modality,
        study_date=record.study_date,
        primary_finding=primary_finding,
        location_count=len(candidates),
        created_at=record.created_at.isoformat(),
    )


def to_detail(record: ReportRecord) -> ReportDetail:
    return ReportDetail(
        **to_summary(record).model_dump(),
        patient_id=record.patient_id,
        patient_age=record.patient_age,
        patient_sex=record.patient_sex,
        body_region=record.body_region,
        vision_model_version=record.vision_model_version,
        candidates=[
            ReportCandidateDetail(
                localization=c["localization"],
                vision_findings=[Finding(**f) for f in c["vision_findings"]],
                copilot_summary=c["copilot_summary"],
                copilot_findings=c["copilot_findings"],
                copilot_limitations=c["copilot_limitations"],
                requires_professional_review=c["requires_professional_review"],
                llm_provider=c["llm_provider"],
                llm_model=c["llm_model"],
            )
            for c in record.candidates()
        ],
    )
