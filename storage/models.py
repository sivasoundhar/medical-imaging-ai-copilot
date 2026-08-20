"""SQLAlchemy ORM models (Day 11, restructured Day 11.1 for multi-
candidate reports). One table: a finished, generated report --
everything needed to redisplay it in the Reports History UI or
regenerate its PDF without re-running inference.

A report covers one or more analyzed candidate locations on the same
patient/study/scan (Section 26 real-world requirement: a single CT read
can cover multiple nodules, not just one). Patient/study metadata is
identical across every candidate in a report, so it stays flat; the
per-candidate findings/explanation vary, so they're stored together as
one JSON list column (`candidates_json`) rather than a normalized child
table -- a report's candidates never change after generation, so this
isn't a case that needs relational structure.
"""
import json
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ReportRecord(Base):
    __tablename__ = "reports"

    # Same value as the AnalysisResponse.analysis_id it was generated
    # from (Section 31) -- one report per analysis, traceable back to it.
    id: Mapped[str] = mapped_column(String, primary_key=True)
    study_id: Mapped[str] = mapped_column(String, index=True)

    # Synthetic demo patient/study metadata (Section 26 "Patient/study
    # metadata") -- never real patient information.
    patient_id: Mapped[str] = mapped_column(String)
    patient_name: Mapped[str] = mapped_column(String)
    patient_age: Mapped[int] = mapped_column()
    patient_sex: Mapped[str] = mapped_column(String)
    modality: Mapped[str] = mapped_column(String)
    body_region: Mapped[str] = mapped_column(String)
    study_date: Mapped[str] = mapped_column(String)

    # One entry per analyzed candidate location, in the order they were
    # added. Each entry: {localization, vision_findings (list[{label,
    # probability}]), heatmap_available, copilot_summary,
    # copilot_findings (list[str]), copilot_limitations (list[str]),
    # requires_professional_review, llm_provider, llm_model}.
    candidates_json: Mapped[str] = mapped_column(Text)

    # Model version metadata (Section 26's report-content requirement).
    vision_model_version: Mapped[str] = mapped_column(String)

    pdf_path: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    def candidates(self) -> list[dict]:
        return json.loads(self.candidates_json)
