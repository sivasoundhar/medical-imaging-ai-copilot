from fastapi.testclient import TestClient

from src.main import app
from src.vision.inference import CTAnalysis, ModelUnavailableError, XrayAnalysis

client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_shape() -> None:
    response = client.get("/health")
    body = response.json()
    assert body["status"] == "ok"
    assert "app_name" in body
    assert "app_env" in body


# --- POST /api/v1/imaging/analyze ---
#
# All tests here mock src.services.imaging_service's imported analyze_xray/
# analyze_ct (from src.vision.inference) — no real checkpoint or GPU
# required, so this suite runs in CI. A manual integration path against
# real checkpoints lives in
# tests/test_api_integration.py (skipped unless real checkpoints exist).

_VALID_PATIENT_FORM = {
    "patient_id": "PT-TEST-001",
    "patient_name": "Test Patient",
    "patient_age": "45",
    "patient_sex": "Male",
}


def _fake_xray_analysis(heatmap_path: str, resized_original_path: str) -> XrayAnalysis:
    return XrayAnalysis(
        prediction="PNEUMONIA",
        probability=0.87,
        class_probabilities={"NORMAL": 0.13, "PNEUMONIA": 0.87},
        heatmap_path=heatmap_path,
        resized_original_path=resized_original_path,
        model_metadata={"architecture": "resnet50", "checkpoint": "model_2d_best.pth"},
        disclaimer="fake disclaimer",
    )


def _fake_ct_analysis(coord_xyz: tuple[float, float, float]) -> CTAnalysis:
    return CTAnalysis(
        prediction="nodule",
        probability=0.91,
        class_probabilities={"non_nodule": 0.09, "nodule": 0.91},
        coord_xyz=coord_xyz,
        model_metadata={"architecture": "nodule_3d_cnn", "checkpoint": "model_3d_best.pth"},
        disclaimer="fake disclaimer",
    )


def test_analyze_xray_success(monkeypatch, tmp_path) -> None:
    # Day 11 fix: heatmap_available is now derived from the heatmap file
    # actually existing/being readable (previously hardcoded True even
    # if the file was missing) -- write a real tiny PNG so this fixture
    # matches what a real Grad-CAM run produces.
    import base64

    heatmap_path = tmp_path / "fake_gradcam.png"
    tiny_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    heatmap_path.write_bytes(tiny_png)
    resized_original_path = tmp_path / "fake_original.png"
    resized_original_path.write_bytes(tiny_png)

    monkeypatch.setattr(
        "src.services.imaging_service.analyze_xray",
        lambda *a, **kw: _fake_xray_analysis(str(heatmap_path), str(resized_original_path)),
    )

    response = client.post(
        "/api/v1/imaging/analyze",
        data={"modality": "xray", **_VALID_PATIENT_FORM},
        files={"file": ("chest.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["study"]["modality"] == "xray"
    assert body["vision"]["findings"] == [{"label": "PNEUMONIA", "probability": 0.87}]
    assert body["vision"]["heatmap_available"] is True
    assert body["vision"]["heatmap_base64"] == base64.b64encode(tiny_png).decode("ascii")
    assert body["vision"]["resized_original_base64"] == base64.b64encode(tiny_png).decode("ascii")
    assert body["vision"]["localization"] is None
    assert body["patient"]["patient_id"] == "PT-TEST-001"
    assert body["safety"]["requires_professional_review"] is True
    assert body["llm"] is None  # not wired until Day 9


def test_analyze_ct_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.services.imaging_service.analyze_ct",
        lambda volume_path, coord_xyz, checkpoint_path: _fake_ct_analysis(coord_xyz),
    )

    response = client.post(
        "/api/v1/imaging/analyze",
        data={
            "modality": "ct",
            "coord_x": "-117.5",
            "coord_y": "25.3",
            "coord_z": "-398.9",
            **_VALID_PATIENT_FORM,
        },
        files={
            "file": ("series.mhd", b"fake-mhd-header", "application/octet-stream"),
            "raw_file": ("series.raw", b"fake-raw-bytes", "application/octet-stream"),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["study"]["modality"] == "ct"
    assert body["vision"]["findings"] == [{"label": "nodule", "probability": 0.91}]
    assert body["vision"]["heatmap_available"] is False  # no 3D Grad-CAM built
    assert body["vision"]["localization"] == "(-117.5, 25.3, -398.9) mm"


def test_analyze_unsupported_modality_returns_422() -> None:
    response = client.post(
        "/api/v1/imaging/analyze",
        data={"modality": "mri", **_VALID_PATIENT_FORM},
        files={"file": ("scan.jpg", b"fake-bytes", "image/jpeg")},
    )
    assert response.status_code == 422
    assert "mri" in response.json()["detail"]


def test_analyze_ct_missing_raw_file_returns_422() -> None:
    response = client.post(
        "/api/v1/imaging/analyze",
        data={
            "modality": "ct",
            "coord_x": "0",
            "coord_y": "0",
            "coord_z": "0",
            **_VALID_PATIENT_FORM,
        },
        files={"file": ("series.mhd", b"fake-mhd-header", "application/octet-stream")},
    )
    assert response.status_code == 422
    assert "raw_file" in response.json()["detail"]


def test_analyze_ct_missing_coordinates_returns_422() -> None:
    response = client.post(
        "/api/v1/imaging/analyze",
        data={"modality": "ct", **_VALID_PATIENT_FORM},
        files={
            "file": ("series.mhd", b"fake-mhd-header", "application/octet-stream"),
            "raw_file": ("series.raw", b"fake-raw-bytes", "application/octet-stream"),
        },
    )
    assert response.status_code == 422
    assert "coord_x" in response.json()["detail"]


def test_analyze_invalid_patient_sex_returns_422() -> None:
    bad_patient = {**_VALID_PATIENT_FORM, "patient_sex": "Unknown"}
    response = client.post(
        "/api/v1/imaging/analyze",
        data={"modality": "xray", **bad_patient},
        files={"file": ("chest.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert response.status_code == 422


def test_analyze_model_unavailable_returns_503_not_a_fabricated_result(monkeypatch) -> None:
    def _raise_unavailable(*a, **kw):
        raise ModelUnavailableError("2D model checkpoint not found: fake/path.pth")

    monkeypatch.setattr("src.services.imaging_service.analyze_xray", _raise_unavailable)

    response = client.post(
        "/api/v1/imaging/analyze",
        data={"modality": "xray", **_VALID_PATIENT_FORM},
        files={"file": ("chest.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 503
    assert "checkpoint not found" in response.json()["detail"]


def test_analyze_invalid_xray_file_returns_422_not_500(monkeypatch) -> None:
    from src.preprocessing.preprocess_2d import InvalidXrayError

    def _raise_invalid(*a, **kw):
        raise InvalidXrayError("Could not read image file")

    monkeypatch.setattr("src.services.imaging_service.analyze_xray", _raise_invalid)

    response = client.post(
        "/api/v1/imaging/analyze",
        data={"modality": "xray", **_VALID_PATIENT_FORM},
        files={"file": ("corrupt.jpg", b"not-a-real-image", "image/jpeg")},
    )

    assert response.status_code == 422
