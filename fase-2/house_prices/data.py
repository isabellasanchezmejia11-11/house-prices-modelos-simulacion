"""Lectura y validación explícita de los CSV de entrenamiento e inferencia."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TRAINING_COLUMNS = {"Id", "SaleCondition", "GrLivArea", "GarageType"}
PREDICTION_COLUMNS = {"GarageType"}


def _read_csv(path: str | Path, description: str) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"No se encontró el archivo de {description}: {source}")
    if not source.is_file():
        raise ValueError(f"La ruta de {description} no es un archivo: {source}")

    data = pd.read_csv(source, keep_default_na=False, na_values=["NA", ""])
    if data.empty:
        raise ValueError(f"El archivo de {description} no contiene registros: {source}")
    return data


def validate_training_data(
    data: pd.DataFrame,
    target_column: str = "SalePrice",
) -> pd.DataFrame:
    """Valida las columnas necesarias y el objetivo antes de iniciar el ajuste."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Los datos de entrenamiento deben ser un DataFrame de pandas.")
    if data.empty:
        raise ValueError("El CSV de entrenamiento no contiene registros.")

    missing_columns = (TRAINING_COLUMNS | {target_column}) - set(data.columns)
    if missing_columns:
        raise ValueError(
            "Faltan columnas requeridas en el CSV de entrenamiento: "
            f"{sorted(missing_columns)}."
        )

    target = pd.to_numeric(data[target_column], errors="coerce")
    if target.isna().any():
        raise ValueError(
            f"'{target_column}' debe contener únicamente valores numéricos no vacíos."
        )
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError(f"'{target_column}' debe contener valores numéricos finitos.")

    return data


def read_training_data(
    path: str | Path,
    target_column: str = "SalePrice",
) -> pd.DataFrame:
    """Carga y valida un CSV que incluye SalePrice."""
    return validate_training_data(_read_csv(path, "entrenamiento"), target_column)


def validate_prediction_data(data: pd.DataFrame) -> pd.DataFrame:
    """Valida un DataFrame con predictores originales y sin variable objetivo."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Los datos de predicción deben ser un DataFrame de pandas.")
    if data.empty:
        raise ValueError("El CSV de predicción no contiene registros.")
    if "SalePrice" in data.columns:
        raise ValueError(
            "El CSV de predicción no debe incluir 'SalePrice'; elimínala antes de predecir."
        )

    missing_columns = PREDICTION_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(
            "Faltan columnas requeridas en el CSV de predicción: "
            f"{sorted(missing_columns)}."
        )
    return data


def read_prediction_data(path: str | Path) -> pd.DataFrame:
    """Carga y valida un CSV de viviendas sin SalePrice."""
    return validate_prediction_data(_read_csv(path, "predicción"))
