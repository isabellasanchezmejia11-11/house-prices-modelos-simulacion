"""Persistencia y carga de versiones identificables del pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import re
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


def save_training_artifacts(
    model: Pipeline,
    output_dir: str | Path,
    version: str,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Guarda el modelo y los informes bajo un identificador de versión único."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
        raise ValueError(
            "La versión solo puede contener letras, números, puntos, guiones y guiones bajos."
        )

    version_dir = Path(output_dir) / version
    version_dir.parent.mkdir(parents=True, exist_ok=True)
    version_dir.mkdir(exist_ok=False)

    model_path = version_dir / "model.joblib"
    metrics_path = version_dir / "metrics.json"
    metadata_path = version_dir / "metadata.json"
    metadata["artifacts"] = {
        "model": model_path.name,
        "metrics": metrics_path.name,
        "metadata": metadata_path.name,
    }
    joblib.dump(model, model_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return model_path, metrics_path, metadata_path


def load_model(model_path: str | Path) -> Pipeline:
    """Carga un pipeline guardado como archivo o dentro de una carpeta de versión."""
    source = Path(model_path)
    if not source.exists():
        raise FileNotFoundError(f"No se encontró el archivo del modelo: {source}")
    if source.is_dir():
        source = source / "model.joblib"
        if not source.is_file():
            raise FileNotFoundError(
                f"No se encontró model.joblib en el directorio de versión: {source.parent}"
            )
    elif not source.is_file():
        raise ValueError(f"La ruta del modelo no es un archivo ni un directorio: {source}")

    try:
        model = joblib.load(source)
    except (EOFError, ImportError, OSError, pickle.UnpicklingError, ValueError) as error:
        raise ValueError(
            f"No se pudo cargar el modelo; el artefacto está dañado o es incompatible: {source}"
        ) from error
    if not isinstance(model, Pipeline):
        raise TypeError(f"El archivo no contiene un pipeline de scikit-learn: {source}")
    return model
