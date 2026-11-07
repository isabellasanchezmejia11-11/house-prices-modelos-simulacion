
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
FASE4 = ROOT / "fase-4"

sys.path.insert(0, str(FASE4))

import registro
from monitoreo import calcular_metricas


def cargar_api():
    ruta = ROOT / "fase-3" / "api.py"
    spec = importlib.util.spec_from_file_location(
        "api_fase4_test", ruta
    )
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    return modulo


@pytest.fixture
def cliente(tmp_path, monkeypatch):
    ruta_csv = tmp_path / "solicitudes.csv"
    monkeypatch.setattr(registro, "REGISTRO_PATH", ruta_csv)

    api = cargar_api()
    app = api.create_app()
    return TestClient(app), ruta_csv


def leer_registros(ruta):
    with ruta.open(encoding="utf-8", newline="") as archivo:
        return list(csv.DictReader(archivo))


def test_registra_prediccion_valida(cliente):
    client, ruta_csv = cliente

    ejemplo = ROOT / "fase-3" / "data" / "example_request.json"
    payload = json.loads(ejemplo.read_text(encoding="utf-8"))

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    filas = leer_registros(ruta_csv)

    assert len(filas) == 1
    assert filas[0]["estado_http"] == "200"
    assert filas[0]["version"] == response.json()["version"]
    assert float(filas[0]["prediccion"]) == pytest.approx(
        response.json()["prediccion"]
    )
    assert filas[0]["fecha_hora"]


def test_registra_prediccion_rechazada(cliente):
    client, ruta_csv = cliente

    response = client.post("/predict", json={})

    assert response.status_code == 422

    filas = leer_registros(ruta_csv)

    assert len(filas) == 1
    assert filas[0]["estado_http"] == "422"
    assert filas[0]["version"] == ""
    assert filas[0]["prediccion"] == ""


def test_calcula_metricas(tmp_path):
    ruta = tmp_path / "solicitudes.csv"

    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow([
            "fecha_hora",
            "estado_http",
            "version",
            "prediccion",
        ])
        writer.writerow(["2026-11-07T10:00:00", 200, "entrega-v1", 100000])
        writer.writerow(["2026-11-07T10:01:00", 200, "entrega-v1", 120000])
        writer.writerow(["2026-11-07T10:02:00", 422, "", ""])

    total, fallidas, porcentaje = calcular_metricas(ruta)

    assert total == 3
    assert fallidas == 1
    assert porcentaje == pytest.approx(33.333333, rel=1e-4)


def test_registro_vacio(tmp_path):
    ruta = tmp_path / "vacio.csv"
    ruta.touch()

    total, fallidas, porcentaje = calcular_metricas(ruta)

    assert total == 0
    assert fallidas == 0
    assert porcentaje is None
