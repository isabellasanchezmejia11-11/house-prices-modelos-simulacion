"""Carga de una versión del modelo y generación de predicciones desde CSV."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


def predict_file(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Predice precios y escribe un CSV con SalePricePrediction y el Id si existe."""
    model = joblib.load(model_path)
    data = pd.read_csv(input_path, keep_default_na=False, na_values=["NA", ""])
    identifiers = data["Id"].copy() if "Id" in data.columns else None
    predictions = model.predict(data)

    output = pd.DataFrame({"SalePricePrediction": predictions})
    if identifiers is not None:
        output.insert(0, "Id", identifiers)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    return destination
