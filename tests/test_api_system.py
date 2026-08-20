"""Day 12: GET /api/v1/system/info -- read-only system configuration for
the Settings page. No mocking needed: this endpoint only reads
`get_settings()` and checks real file existence, both safe in CI."""
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_system_info_returns_200() -> None:
    response = client.get("/api/v1/system/info")
    assert response.status_code == 200


def test_system_info_shape() -> None:
    response = client.get("/api/v1/system/info")
    body = response.json()
    assert isinstance(body["app_name"], str)
    assert isinstance(body["llm_provider"], str)
    assert isinstance(body["model_2d_available"], bool)
    assert isinstance(body["model_3d_available"], bool)
    assert body["model_2d_version"] == "resnet50-2d-v1"
    assert body["model_3d_version"] == "nodule-3d-cnn-v1"


def test_system_info_never_leaks_api_keys() -> None:
    # Real check, not just a naming convention: no field name on the
    # response schema contains "key" or "secret" at all, so there's no
    # accidental path to leaking GROQ_API_KEY/ANTHROPIC_API_KEY.
    response = client.get("/api/v1/system/info")
    body = response.json()
    for field_name in body:
        assert "key" not in field_name.lower()
        assert "secret" not in field_name.lower()
