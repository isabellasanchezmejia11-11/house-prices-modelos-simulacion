"""Entrena el modelo desde un CSV con SalePrice."""

from __future__ import annotations

import argparse

import pandas as pd

from house_prices.model import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Ruta al CSV de entrenamiento.")
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directorio para guardar la versión entrenada (default: models).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.data, keep_default_na=False, na_values=["NA", ""])
    model_path, metrics = train_model(
        data,
        args.models_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Modelo guardado: {model_path}")
    print(
        "Métricas de evaluación "
        f"(MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.4f})"
    )


if __name__ == "__main__":
    main()
