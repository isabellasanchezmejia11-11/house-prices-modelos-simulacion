"""Carga de una versión del modelo y generación de predicciones desde CSV."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from house_prices.data import read_prediction_data, validate_prediction_data
from house_prices.storage import load_model

OPTIONAL_INPUT_COLUMNS = {"Id", "SaleType", "SaleCondition"}


def predict_file(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Predice precios desde un pipeline guardado y escribe el CSV de resultados."""
    model = load_model(model_path)
    data = read_prediction_data(input_path)
    return write_predictions(model, data, output_path)


def write_predictions(
    model: Pipeline,
    data: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Valida un DataFrame y persiste las predicciones y su identificador opcional."""
    validate_prediction_data(data)
    identifiers = data["Id"].copy() if "Id" in data.columns else None
    model_data = _prepare_model_input(model, data)
    predictions = model.predict(model_data)
    if len(predictions) != len(data):
        raise ValueError("El modelo devolvió una cantidad de predicciones inesperada.")
    if not np.isfinite(predictions).all():
        raise ValueError("El modelo produjo predicciones no finitas.")

    output = pd.DataFrame({"SalePricePrediction": predictions})
    if identifiers is not None:
        output.insert(0, "Id", identifiers)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        output.to_csv(temporary_path, index=False)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _prepare_model_input(model: Pipeline, data: pd.DataFrame) -> pd.DataFrame:
    """Valida tipos y adapta las columnas al esquema aprendido por el pipeline."""
    schema = getattr(model, "feature_names_in_", None)
    if schema is None:
        raise ValueError("El modelo no contiene un esquema de columnas para validar la entrada.")
    schema = [str(column) for column in schema]

    column_transformer = model.named_steps.get("preprocessing")
    if column_transformer is None or not hasattr(column_transformer, "transformers_"):
        raise ValueError("El modelo no contiene el preprocesamiento esperado.")

    numeric_columns: set[str] = set()
    categorical_columns: set[str] = set()
    for name, _, columns in column_transformer.transformers_:
        if name == "numeric":
            numeric_columns.update(str(column) for column in columns)
        elif name == "categorical":
            categorical_columns.update(str(column) for column in columns)

    required_columns = set(schema) - OPTIONAL_INPUT_COLUMNS
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            "Faltan columnas requeridas por el modelo: "
            f"{sorted(missing_columns)}."
        )

    model_data = data.copy()
    for column in OPTIONAL_INPUT_COLUMNS.intersection(schema):
        if column not in model_data.columns:
            model_data[column] = np.nan

    for column in numeric_columns.intersection(model_data.columns):
        original = model_data[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            raise ValueError(
                f"La columna numérica '{column}' contiene valores no numéricos."
            )
        if not np.isfinite(converted.dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"La columna numérica '{column}' contiene valores no finitos.")
        model_data[column] = converted

    for column in categorical_columns.intersection(model_data.columns):
        values = model_data[column].dropna()
        if not values.map(lambda value: isinstance(value, str)).all():
            raise ValueError(
                f"La columna categórica '{column}' debe contener texto o valores nulos."
            )

    return model_data.reindex(columns=schema)
