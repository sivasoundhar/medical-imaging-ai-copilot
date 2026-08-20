"""Repository functions for `ReportRecord` (Day 11). Plain sync
functions -- async endpoints call these via `asyncio.to_thread()`
rather than this module knowing anything about asyncio itself.
"""
import json
from dataclasses import dataclass

from storage.database import get_session
from storage.models import ReportRecord


@dataclass
class NewReport:
    """Plain data the caller assembles before persisting -- keeps this
    module decoupled from the Pydantic schemas in `src/schemas/`."""

    id: str
    study_id: str
    patient_id: str
    patient_name: str
    patient_age: int
    patient_sex: str
    modality: str
    body_region: str
    study_date: str
    candidates: list[dict]  # see storage/models.py's ReportRecord docstring for shape
    vision_model_version: str
    pdf_path: str


def save_report(data: NewReport) -> ReportRecord:
    with get_session() as session:
        record = ReportRecord(
            id=data.id,
            study_id=data.study_id,
            patient_id=data.patient_id,
            patient_name=data.patient_name,
            patient_age=data.patient_age,
            patient_sex=data.patient_sex,
            modality=data.modality,
            body_region=data.body_region,
            study_date=data.study_date,
            candidates_json=json.dumps(data.candidates),
            vision_model_version=data.vision_model_version,
            pdf_path=data.pdf_path,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        session.expunge(record)  # detach so it's usable after the session closes
        return record


def get_report(report_id: str) -> ReportRecord | None:
    with get_session() as session:
        record = session.get(ReportRecord, report_id)
        if record is not None:
            session.expunge(record)
        return record


def list_reports(limit: int = 100) -> list[ReportRecord]:
    with get_session() as session:
        records = (
            session.query(ReportRecord)
            .order_by(ReportRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        for r in records:
            session.expunge(r)
        return records


def delete_report(report_id: str) -> str | None:
    """Day 12: permanently deletes a report record. Returns the deleted
    row's `pdf_path` (captured before the delete, since SQLAlchemy
    expires instance attributes on commit -- reading it afterwards would
    re-query a now-nonexistent row) so the caller can also remove the
    PDF file from disk. This function only touches the DB row,
    deliberately, so a filesystem failure never leaves a half-deleted DB
    state. Returns None if no report with that id existed (caller maps
    that to 404)."""
    with get_session() as session:
        record = session.get(ReportRecord, report_id)
        if record is None:
            return None
        pdf_path = record.pdf_path
        session.delete(record)
        session.commit()
        return pdf_path
