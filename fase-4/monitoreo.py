
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from registro import REGISTRO_PATH


def calcular_metricas(ruta: Path) -> tuple[int, int, float | None]:
    if not ruta.exists():
        return 0, 0, None

    total = 0
    fallidas = 0

    with ruta.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as archivo:
        lector = csv.DictReader(archivo)

        for fila in lector:
            if not fila.get("estado_http"):
                continue

            codigo = int(fila["estado_http"])
            total += 1

            if 400 <= codigo <= 599:
                fallidas += 1

    porcentaje = (
        fallidas / total * 100
        if total > 0
        else None
    )

    return total, fallidas, porcentaje


def main():
    parser = argparse.ArgumentParser(
        description="Indicadores de operación de la API House Prices"
    )

    parser.add_argument(
        "--registro",
        type=Path,
        default=REGISTRO_PATH,
        help="Ruta del archivo CSV de solicitudes.",
    )

    args = parser.parse_args()

    total, fallidas, porcentaje = calcular_metricas(
        args.registro
    )

    print("=== Monitoreo de la API House Prices ===")
    print(f"Total de solicitudes: {total}")
    print(f"Solicitudes fallidas: {fallidas}")

    if porcentaje is None:
        print("Porcentaje de fallos: sin solicitudes")
    else:
        print(f"Porcentaje de fallos: {porcentaje:.1f} %")


if __name__ == "__main__":
    main()
