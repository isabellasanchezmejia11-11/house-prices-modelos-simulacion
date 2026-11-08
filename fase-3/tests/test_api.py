import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
API_FILE = ROOT / "fase-3" / "api.py"
EXAMPLE_FILE = ROOT / "fase-3" / "data" / "example_request.json"

spec = importlib.util.spec_from_file_location("fase3_api", API_FILE)
api_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api_module
spec.loader.exec_module(api_module)


@pytest.fixture
def client(tmp_path, monkeypatch):
    registro = sys.modules["registro"]
    monkeypatch.setattr(registro, "REGISTRO_PATH", tmp_path / "solicitudes.csv")
    with TestClient(api_module.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["estado"] == "ok"


def test_predict_valid(client):
    with open(EXAMPLE_FILE, encoding="utf-8") as file:
        payload = json.load(file)

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediccion" in response.json()
    assert "version" in response.json()


def test_predict_missing_required_field(client):
    with open(EXAMPLE_FILE, encoding="utf-8") as file:
        payload = json.load(file)

    model = api_module.load_model(
        api_module.app.state.settings.models_dir
        / api_module.app.state.settings.model_version
    )

    required = [
        str(column)
        for column in model.feature_names_in_
        if str(column) not in api_module.OPTIONAL_INPUT_COLUMNS
    ]

    assert required, "No se encontraron campos obligatorios"
    payload.pop(required[0], None)

    response = client.post("/predict", json=payload)

    assert response.status_code in (400, 422)
    assert required[0] in str(response.json())
