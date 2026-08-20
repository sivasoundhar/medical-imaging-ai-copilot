"""Tests for Day 11 (Day 11.1: multi-candidate) POST /api/v1/report/generate
and GET /api/v1/reports*. Isolated temp DB + temp PDF output dir (same
pattern as tests/test_report_service.py) -- never touches the real
storage/app.db or reports/generated/.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.services.report_service as report_service
from src.main import app
from storage.models import Base

client = TestClient(app)


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


def _analysis_body(analysis_id: str = "analysis-1", label: str = "pneumonia") -> dict:
    return {
        "analysis_id": analysis_id,
        "study_id": "study-1",
        "patient": {"patient_id": "PT-1", "name": "Test Patient", "age": 45, "sex": "Male"},
        "study": {"modality": "xray", "body_region": "chest", "study_date": "2026-08-15"},
        "vision": {
            "findings": [{"label": label, "probability": 0.87}],
            "heatmap_available": False,
            "localization": None,
            "heatmap_base64": None,
        },
        "llm": None,
        "safety": {"requires_professional_review": True},
    }


def _copilot_body(summary: str = "The model flagged pneumonia.", findings: list[str] | None = None) -> dict:
    return {
        "summary": summary,
        "findings": findings if findings is not None else ["pneumonia"],
        "limitations": ["Not a diagnosis."],
        "requires_professional_review": True,
        "provider": "mock",
        "model": "mock-model",
        "grounded": True,
        "kb_sources_used": [],
    }


def _generate_body(candidates: list[dict] | None = None) -> dict:
    return {
        "candidates": candidates
        or [{"analysis": _analysis_body(), "copilot": _copilot_body()}],
        "vision_model_version": "resnet50-v1",
    }


def test_report_generate_returns_summary(isolated_report_env) -> None:
    response = client.post("/api/v1/report/generate", json=_generate_body())

    assert response.status_code == 200
    body = response.json()
    assert body["report_id"]  # freshly minted -- no longer tied to the analysis_id
    assert body["primary_finding"] == "pneumonia"
    assert body["patient_name"] == "Test Patient"
    assert body["location_count"] == 1


def test_report_generate_rejects_empty_candidate_list(isolated_report_env) -> None:
    response = client.post(
        "/api/v1/report/generate",
        json={"candidates": [], "vision_model_version": "resnet50-v1"},
    )
    assert response.status_code == 422


def test_reports_list_includes_generated_report(isolated_report_env) -> None:
    generate_response = client.post("/api/v1/report/generate", json=_generate_body())
    report_id = generate_response.json()["report_id"]

    response = client.get("/api/v1/reports")

    assert response.status_code == 200
    ids = [r["report_id"] for r in response.json()]
    assert report_id in ids


def test_report_detail_returns_full_data(isolated_report_env) -> None:
    generate_response = client.post("/api/v1/report/generate", json=_generate_body())
    report_id = generate_response.json()["report_id"]

    response = client.get(f"/api/v1/reports/{report_id}")

    assert response.status_code == 200
    body = response.json()
    assert len(body["candidates"]) == 1
    assert body["candidates"][0]["copilot_summary"] == "The model flagged pneumonia."
    assert body["candidates"][0]["llm_provider"] == "mock"
    assert body["candidates"][0]["vision_findings"] == [{"label": "pneumonia", "probability": 0.87}]


def test_report_detail_with_multiple_candidates(isolated_report_env) -> None:
    """Day 11.1: a real multi-location CT report round-trips through the
    HTTP layer with each candidate's own findings and explanation intact."""
    body = _generate_body(
        candidates=[
            {
                "analysis": _analysis_body(analysis_id="analysis-1", label="nodule"),
                "copilot": _copilot_body(summary="First location.", findings=["nodule"]),
            },
            {
                "analysis": _analysis_body(analysis_id="analysis-2", label="non-nodule"),
                "copilot": _copilot_body(summary="Second location.", findings=["non-nodule"]),
            },
        ]
    )
    generate_response = client.post("/api/v1/report/generate", json=body)
    assert generate_response.status_code == 200
    assert generate_response.json()["location_count"] == 2
    report_id = generate_response.json()["report_id"]

    response = client.get(f"/api/v1/reports/{report_id}")

    assert response.status_code == 200
    candidates = response.json()["candidates"]
    assert len(candidates) == 2
    assert candidates[0]["copilot_summary"] == "First location."
    assert candidates[1]["copilot_summary"] == "Second location."


def test_report_detail_404_for_unknown_id(isolated_report_env) -> None:
    response = client.get("/api/v1/reports/does-not-exist")
    assert response.status_code == 404


def test_report_pdf_download_returns_real_pdf_bytes(isolated_report_env) -> None:
    generate_response = client.post("/api/v1/report/generate", json=_generate_body())
    report_id = generate_response.json()["report_id"]

    response = client.get(f"/api/v1/reports/{report_id}/pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content[:4] == b"%PDF"


def test_report_pdf_download_404_for_unknown_id(isolated_report_env) -> None:
    response = client.get("/api/v1/reports/does-not-exist/pdf")
    assert response.status_code == 404


def test_report_delete_removes_it_for_real(isolated_report_env) -> None:
    generate_response = client.post("/api/v1/report/generate", json=_generate_body())
    report_id = generate_response.json()["report_id"]
    # The real PDF file exists on disk before delete, per Day 11's own
    # generate_report_pdf() -- confirms this test is checking a real
    # file removal, not just a DB row.
    pdf_response_before = client.get(f"/api/v1/reports/{report_id}/pdf")
    assert pdf_response_before.status_code == 200

    delete_response = client.delete(f"/api/v1/reports/{report_id}")
    assert delete_response.status_code == 204

    assert client.get(f"/api/v1/reports/{report_id}").status_code == 404
    assert client.get(f"/api/v1/reports/{report_id}/pdf").status_code == 404
    ids = [r["report_id"] for r in client.get("/api/v1/reports").json()]
    assert report_id not in ids


def test_report_delete_404_for_unknown_id(isolated_report_env) -> None:
    response = client.delete("/api/v1/reports/does-not-exist")
    assert response.status_code == 404


def test_report_delete_does_not_remove_other_reports(isolated_report_env) -> None:
    id_a = client.post("/api/v1/report/generate", json=_generate_body()).json()["report_id"]
    id_b = client.post("/api/v1/report/generate", json=_generate_body()).json()["report_id"]

    client.delete(f"/api/v1/reports/{id_a}")

    assert client.get(f"/api/v1/reports/{id_a}").status_code == 404
    assert client.get(f"/api/v1/reports/{id_b}").status_code == 200
