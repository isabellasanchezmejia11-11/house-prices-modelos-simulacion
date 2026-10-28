"""Entrena y evalúa el modelo desde un CSV etiquetado."""

from __future__ import annotations

import argparse

from house_prices.data import read_training_data
from house_prices.model import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Ruta al CSV de entrenamiento.")
    parser.add_argument(
        "--target",
        required=True,
        help="Nombre de la columna objetivo numérica (para Ames: SalePrice).",
    )
    parser.add_argument("--output-dir", required=True, help="Directorio para artefactos.")
    parser.add_argument("--version", required=True, help="Identificador único de versión.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fracción de registros reservada para evaluación (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla no negativa para reproducir la partición (default: 42).",
    )
    args = parser.parse_args()

    data = read_training_data(args.data, args.target)
    artifacts = train_model(
        data,
        args.output_dir,
        version=args.version,
        target_column=args.target,
        source_data=args.data,
        test_size=args.test_size,
        random_state=args.seed,
    )
    model_metrics = artifacts.metrics["models"]["linear_regression"]
    baseline_metrics = artifacts.metrics["models"]["dummy_regressor"]["metrics"]
    print(f"Versión: {args.version}")
    print(
        "Regresión lineal "
        f"(MAE={model_metrics['mae']:.2f}, RMSE={model_metrics['rmse']:.2f}, "
        f"R²={model_metrics['r2']:.4f})"
    )
    print(
        "DummyRegressor "
        f"(MAE={baseline_metrics['mae']:.2f}, RMSE={baseline_metrics['rmse']:.2f}, "
        f"R²={baseline_metrics['r2']:.4f})"
    )
    print(f"Modelo: {artifacts.model_path}")
    print(f"Métricas: {artifacts.metrics_path}")
    print(f"Metadatos: {artifacts.metadata_path}")


if __name__ == "__main__":
    main()
