"""Manual integration path for the Day 8 `/api/v1/imaging/analyze`
endpoint, against REAL trained checkpoints and REAL data — not mocked
(tests/test_api.py covers the mocked/CI path). Per Day 8's "Add a manual
integration path for real models."

Each test skips itself (not a hard failure) when the checkpoint/data it
needs isn't present — both `training/checkpoints/*.pth` and `Data/` are
gitignored, so a fresh clone/CI runner won't have them. Run explicitly:

    pytest tests/test_api_integration.py -v
"""
import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.config import get_settings
from src.main import app

client = TestClient(app)
settings = get_settings()

_MODEL_2D = Path(settings.model_2d_checkpoint_path)
_MODEL_3D = Path(settings.model_3d_checkpoint_path)
_XRAY_DIR = Path("Data/chest_xray/chest_xray/test")  # real layout nests an extra chest_xray/ level
_CT_DIR = Path("Data/subset0")
_CANDIDATES_CSV = Path("Data/candidates.csv")

_VALID_PATIENT_FORM = {
    "patient_id": "PT-INTEGRATION-001",
    "patient_name": "Integration Test Patient",
    "patient_age": "50",
    "patient_sex": "Female",
}


def _find_one_real_xray() -> Path | None:
    if not _XRAY_DIR.exists():
        return None
    # Skip macOS AppleDouble resource-fork files ("._name.jpeg") -- not
    # real images. dataset_2d.py's own splits never see these (it globs
    # the class dirs directly, not the dataset root's __MACOSX/ folder),
    # but this test's search should stay defensive too.
    return next(
        (p for p in _XRAY_DIR.rglob("*.jpeg") if not p.name.startswith("._")), None
    )


def _find_one_real_ct_candidate() -> tuple[Path, tuple[float, float, float]] | None:
    if not (_CT_DIR.exists() and _CANDIDATES_CSV.exists()):
        return None
    available = {p.stem for p in _CT_DIR.glob("*.mhd")}
    with open(_CANDIDATES_CSV) as f:
        for row in csv.DictReader(f):
            if row["seriesuid"] in available:
                return (
                    _CT_DIR / f"{row['seriesuid']}.mhd",
                    (float(row["coordX"]), float(row["coordY"]), float(row["coordZ"])),
                )
    return None


@pytest.mark.skipif(not _MODEL_2D.exists(), reason="Real model_2d checkpoint not present")
def test_real_xray_end_to_end() -> None:
    xray_path = _find_one_real_xray()
    if xray_path is None:
        pytest.skip("No real X-ray image found under Data/chest_xray")

    with open(xray_path, "rb") as f:
        response = client.post(
            "/api/v1/imaging/analyze",
            data={"modality": "xray", **_VALID_PATIENT_FORM},
            files={"file": (xray_path.name, f, "image/jpeg")},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    finding = body["vision"]["findings"][0]
    assert finding["label"] in ("NORMAL", "PNEUMONIA")
    assert 0.0 <= finding["probability"] <= 1.0
    assert body["vision"]["heatmap_available"] is True


@pytest.mark.skipif(not _MODEL_3D.exists(), reason="Real model_3d checkpoint not present")
def test_real_ct_end_to_end() -> None:
    found = _find_one_real_ct_candidate()
    if found is None:
        pytest.skip("No real CT volume + candidates.csv row found under Data/")
    mhd_path, (x, y, z) = found
    raw_path = mhd_path.with_suffix(".raw")

    with open(mhd_path, "rb") as mhd_f, open(raw_path, "rb") as raw_f:
        response = client.post(
            "/api/v1/imaging/analyze",
            data={
                "modality": "ct",
                "coord_x": str(x),
                "coord_y": str(y),
                "coord_z": str(z),
                **_VALID_PATIENT_FORM,
            },
            files={
                "file": (mhd_path.name, mhd_f, "application/octet-stream"),
                "raw_file": (raw_path.name, raw_f, "application/octet-stream"),
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    finding = body["vision"]["findings"][0]
    assert finding["label"] in ("nodule", "non_nodule")
    assert 0.0 <= finding["probability"] <= 1.0
    assert body["vision"]["heatmap_available"] is False
    assert body["vision"]["localization"] == f"({x:.1f}, {y:.1f}, {z:.1f}) mm"
