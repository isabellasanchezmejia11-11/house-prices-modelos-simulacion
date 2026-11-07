from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

REGISTRO_PATH = (
    Path(__file__).resolve().parent
    / "registros"
    / "solicitudes.csv"
)

CAMPOS = [
    "fecha_hora",
    "estado_http",
    "version",
    "prediccion",
]


def guardar_solicitud(
    estado_http: int,
    version: str | None = None,
    prediccion: float | None = None,
) -> None:
    REGISTRO_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nuevo_archivo = (
        not REGISTRO_PATH.exists()
        or REGISTRO_PATH.stat().st_size == 0
    )

    with REGISTRO_PATH.open(
        mode="a",
        newline="",
        encoding="utf-8",
    ) as archivo:
        writer = csv.DictWriter(
            archivo,
            fieldnames=CAMPOS,
        )

        if nuevo_archivo:
            writer.writeheader()

        writer.writerow({
            "fecha_hora": datetime.now(
                timezone.utc
            ).isoformat(),
            "estado_http": estado_http,
            "version": version or "",
            "prediccion": (
                prediccion
                if prediccion is not None
                else ""
            ),
        })
