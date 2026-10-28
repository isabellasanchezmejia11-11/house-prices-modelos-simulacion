"""Genera predicciones con una versión guardada del modelo."""

from __future__ import annotations

import argparse

import pandas as pd

from house_prices.prediction import predict_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Ruta a model.joblib o al directorio de una versión del modelo.",
    )
    parser.add_argument(
        "--input",
        "--data",
        dest="input_path",
        required=True,
        help="Ruta al CSV no vacío de viviendas sin SalePrice.",
    )
    parser.add_argument("--output", required=True, help="Ruta del CSV de predicciones.")
    args = parser.parse_args()

    try:
        output_path = predict_file(args.model, args.input_path, args.output)
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as error:
        parser.error(str(error))
    print(f"Predicciones guardadas: {output_path}")


if __name__ == "__main__":
    main()
