"""Genera predicciones con una versión guardada del modelo."""

from __future__ import annotations

import argparse

from house_prices.prediction import predict_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Ruta al archivo .joblib.")
    parser.add_argument("--data", required=True, help="CSV de viviendas sin SalePrice.")
    parser.add_argument("--output", required=True, help="Ruta del CSV de predicciones.")
    args = parser.parse_args()

    output_path = predict_file(args.model, args.data, args.output)
    print(f"Predicciones guardadas: {output_path}")


if __name__ == "__main__":
    main()
