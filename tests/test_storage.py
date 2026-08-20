"""Tests for the Day 11 SQLite storage layer (storage/). Uses an
isolated temp database per test (monkeypatching
`storage.repositories.get_session` -- the name as imported *there*,
not `storage.database.get_session`, since Python binds `from x import
y` at import time) -- never touches the real `storage/app.db`.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from storage.models import Base
from storage.repositories import NewReport, delete_report, get_report, list_reports, save_report


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path/'test.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr("storage.repositories.get_session", TestSessionLocal)
    yield


def _candidate(**overrides) -> dict:
    defaults = dict(
        localization=None,
        vision_findings=[{"label": "pneumonia", "probability": 0.87}],
        heatmap_available=True,
        copilot_summary="The model flagged pneumonia.",
        copilot_findings=["pneumonia"],
        copilot_limitations=["not a diagnosis"],
        requires_professional_review=True,
        llm_provider="mock",
        llm_model="mock-model",
    )
    defaults.update(overrides)
    return defaults


def _new_report(**overrides) -> NewReport:
    defaults = dict(
        id="analysis-1",
        study_id="study-1",
        patient_id="PT-1",
        patient_name="Test Patient",
        patient_age=40,
        patient_sex="Male",
        modality="xray",
        body_region="chest",
        study_date="2026-08-15",
        candidates=[_candidate()],
        vision_model_version="resnet50-v1",
        pdf_path="reports/generated/analysis-1.pdf",
    )
    defaults.update(overrides)
    return NewReport(**defaults)


def test_save_and_get_report_round_trips_all_fields(isolated_db):
    save_report(_new_report())

    record = get_report("analysis-1")

    assert record is not None
    assert record.patient_name == "Test Patient"
    candidates = record.candidates()
    assert len(candidates) == 1
    assert candidates[0]["vision_findings"] == [{"label": "pneumonia", "probability": 0.87}]
    assert candidates[0]["copilot_findings"] == ["pneumonia"]
    assert candidates[0]["copilot_limitations"] == ["not a diagnosis"]
    assert candidates[0]["requires_professional_review"] is True


def test_save_and_get_report_round_trips_multiple_candidates(isolated_db):
    save_report(
        _new_report(
            id="multi-1",
            candidates=[
                _candidate(localization="RUL", copilot_findings=["nodule"]),
                _candidate(localization="LLL", copilot_findings=["non-nodule"]),
            ],
        )
    )

    record = get_report("multi-1")

    assert record is not None
    candidates = record.candidates()
    assert len(candidates) == 2
    assert candidates[0]["localization"] == "RUL"
    assert candidates[1]["localization"] == "LLL"


def test_get_report_returns_none_for_missing_id(isolated_db):
    assert get_report("does-not-exist") is None


def test_list_reports_orders_newest_first(isolated_db):
    import time

    save_report(_new_report(id="a"))
    time.sleep(0.01)  # ensure a distinct created_at than "b"/"c" -- avoids a timestamp-tie flake
    save_report(_new_report(id="b"))
    time.sleep(0.01)
    save_report(_new_report(id="c"))

    ids = [r.id for r in list_reports()]

    assert ids[0] == "c"  # newest first
    assert ids[-1] == "a"  # oldest last
    assert set(ids) == {"a", "b", "c"}


def test_list_reports_respects_limit(isolated_db):
    for i in range(5):
        save_report(_new_report(id=f"report-{i}"))

    assert len(list_reports(limit=2)) == 2


def test_delete_report_removes_it_and_returns_pdf_path(isolated_db):
    save_report(_new_report(id="to-delete", pdf_path="reports/generated/to-delete.pdf"))

    pdf_path = delete_report("to-delete")

    assert pdf_path == "reports/generated/to-delete.pdf"
    assert get_report("to-delete") is None


def test_delete_report_returns_none_for_missing_id(isolated_db):
    assert delete_report("does-not-exist") is None


def test_delete_report_does_not_affect_other_reports(isolated_db):
    save_report(_new_report(id="keep-me"))
    save_report(_new_report(id="delete-me"))

    delete_report("delete-me")

    assert get_report("keep-me") is not None
    assert get_report("delete-me") is None
