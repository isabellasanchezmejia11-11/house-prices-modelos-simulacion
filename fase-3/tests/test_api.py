import hashlib
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


@pytest.mark.parametrize("version", ["entrega-v1", "entrega-v2"])
def test_predict_matches_delivery_reference(client, version):
    reference = json.loads(
        (ROOT / "fase-3/data/prediction_reference.json").read_text(encoding="utf-8")
    )
    input_bytes = (ROOT / reference["input_file"]).read_bytes().replace(bytes([13, 10]), bytes([10]))
    assert hashlib.sha256(input_bytes).hexdigest() == reference["input_sha256"], (
        "La entrada conocida cambió; revisar la referencia antes de actualizarla."
    )
    app = api_module.create_app(api_module.Settings(
        models_dir=ROOT / "fase-2/models", model_version=version,
    ))
    with TestClient(app) as reference_client:
        response = reference_client.post("/predict", json=json.loads(input_bytes))

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["version"] == version
    assert result["prediccion"] == pytest.approx(
        reference["predictions"][version],
        rel=reference["relative_tolerance"],
        abs=reference["absolute_tolerance"],
    )
