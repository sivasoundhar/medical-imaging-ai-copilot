"""Tests for Day 11's POST /api/v1/copilot/report and /copilot/ask.

Mocked at `src.main.get_llm_providers` (returns a `[MockLLMProvider]`) --
no real network calls, matching the project's established convention
(Day 8/9's "mocked providers for CI"). Real end-to-end verification
against a live server + real Ollama was done manually for this feature
(see PROGRESS_LOG.md Day 11 entry).
"""
import json

from fastapi.testclient import TestClient

from src.llm.mock_provider import MockLLMProvider
from src.main import app

client = TestClient(app)

_GOOD_REPORT_JSON = json.dumps(
    {
        "summary": "The model flagged pneumonia with a probability of 0.87.",
        "findings": ["pneumonia"],
        "limitations": ["A single X-ray and model score cannot confirm a diagnosis."],
        "requires_professional_review": True,
    }
)

_REQUEST_BODY = {
    "findings": [{"label": "pneumonia", "probability": 0.87}],
    "localization": "right_lower_lung",
    "modality": "xray",
}


def test_copilot_report_returns_grounded_response(monkeypatch) -> None:
    monkeypatch.setattr("src.main.get_llm_providers", lambda: [MockLLMProvider(response=_GOOD_REPORT_JSON)])

    response = client.post("/api/v1/copilot/report", json=_REQUEST_BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["findings"] == ["pneumonia"]
    assert body["grounded"] is True
    assert body["provider"] == "mock"
    assert body["requires_professional_review"] is True


def test_copilot_report_rejects_ungrounded_response_as_422(monkeypatch) -> None:
    bad_json = json.dumps(
        {
            "summary": "The model flagged pneumonia and pleural effusion.",
            "findings": ["pneumonia", "pleural effusion"],  # invented, not in request findings
            "limitations": ["N/A"],
            "requires_professional_review": True,
        }
    )
    monkeypatch.setattr("src.main.get_llm_providers", lambda: [MockLLMProvider(response=bad_json)])

    response = client.post("/api/v1/copilot/report", json=_REQUEST_BODY)

    assert response.status_code == 422
    assert "pleural effusion" in response.json()["detail"]


def test_copilot_report_returns_503_when_provider_not_configured(monkeypatch) -> None:
    from src.llm.base import LLMConfigurationError

    def _raise(*a, **kw):
        raise LLMConfigurationError("LLM_PROVIDER=groq requires GROQ_API_KEY to be set.")

    monkeypatch.setattr("src.main.get_llm_providers", _raise)

    response = client.post("/api/v1/copilot/report", json=_REQUEST_BODY)

    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_copilot_ask_returns_grounded_answer(monkeypatch) -> None:
    monkeypatch.setattr("src.main.get_llm_providers", lambda: [MockLLMProvider(response=_GOOD_REPORT_JSON)])

    response = client.post(
        "/api/v1/copilot/ask",
        json={**_REQUEST_BODY, "question": "Why did the model flag this image?"},
    )

    assert response.status_code == 200
    assert response.json()["grounded"] is True


def test_copilot_ask_rejects_empty_question_as_422(monkeypatch) -> None:
    monkeypatch.setattr("src.main.get_llm_providers", lambda: [MockLLMProvider(response=_GOOD_REPORT_JSON)])

    response = client.post("/api/v1/copilot/ask", json={**_REQUEST_BODY, "question": "   "})

    assert response.status_code == 422
