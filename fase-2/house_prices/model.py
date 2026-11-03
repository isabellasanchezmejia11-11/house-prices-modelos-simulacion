"""Construcción, entrenamiento y persistencia del modelo de regresión."""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from house_prices.data import validate_training_data
from house_prices.pipeline import build_model
from house_prices.storage import save_training_artifacts


@dataclass(frozen=True)
class TrainingArtifacts:
    model_path: Path
    metrics_path: Path
    metadata_path: Path
    metrics: dict[str, Any]


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Calcula métricas sobre datos no usados para ajustar el modelo."""
    predictions = model.predict(X_test)

    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }


def train_model(
    data: pd.DataFrame,
    output_dir: str | Path,
    version: str,
    target_column: str = "SalePrice",
    source_data: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingArtifacts:
    """Evalúa modelo y baseline y guarda los artefactos de la versión."""
    validate_training_data(data, target_column)

    if len(data) < 3:
        raise ValueError(
            "Se requieren al menos tres filas para entrenar y evaluar."
        )

    if not 0 < test_size < 1:
        raise ValueError("test_size debe ser mayor que 0 y menor que 1.")

    if random_state < 0:
        raise ValueError("random_state debe ser un entero no negativo.")

    target = pd.to_numeric(data[target_column], errors="raise")
    features = data.drop(columns=[target_column])

    # Separar antes de ajustar cualquier transformación estadística.
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    required_outlier_columns = {"Id", "SaleCondition", "GrLivArea"}
    missing_outlier_columns = required_outlier_columns - set(X_train.columns)

    if missing_outlier_columns:
        raise ValueError(
            "El CSV debe incluir las columnas requeridas para tratar los "
            f"outliers documentados: {sorted(missing_outlier_columns)}."
        )

    # Excluir los outliers documentados únicamente del entrenamiento.
    outlier_mask = (
        X_train["Id"].isin([524, 1299])
        & X_train["SaleCondition"].eq("Partial")
        & pd.to_numeric(
            X_train["GrLivArea"],
            errors="coerce",
        ).gt(4000)
    )

    training_outlier_ids = [
        int(value)
        for value in X_train["Id"].loc[outlier_mask]
    ]

    X_train = X_train.loc[~outlier_mask].copy()
    y_train = y_train.loc[~outlier_mask]

    # Ajustar el pipeline completo solo con entrenamiento.
    evaluation_model = build_model()
    evaluation_model.fit(X_train, y_train)

    # Evaluar el modelo base sobre la misma partición.
    baseline_model = DummyRegressor(strategy="mean")
    baseline_model.fit(X_train, y_train)

    metrics = {
        "target": target_column,
        "source_data": source_data,
        "split": {
            "test_size": test_size,
            "seed": random_state,
            "training_rows": len(X_train),
            "test_rows": len(X_test),
            "training_outlier_ids_removed": training_outlier_ids,
        },
        "models": {
            "linear_regression": evaluate_model(
                evaluation_model,
                X_test,
                y_test,
            ),
            "dummy_regressor": {
                "strategy": "mean",
                "metrics": evaluate_model(
                    baseline_model,
                    X_test,
                    y_test,
                ),
            },
        },
    }

    # Identificar el código utilizado, incluso sin historial de Git.
    application_dir = Path(__file__).resolve().parent.parent

    code_files = sorted(
        list((application_dir / "house_prices").glob("*.py"))
        + [
            application_dir / "train.py",
            application_dir / "predict.py",
        ],
        key=lambda path: path.relative_to(application_dir).as_posix(),
    )

    code_hasher = hashlib.sha256()

    for code_file in code_files:
        relative_path = code_file.relative_to(application_dir).as_posix()
        content = code_file.read_bytes()

        code_hasher.update(relative_path.encode("utf-8") + b"\0")
        code_hasher.update(str(len(content)).encode("ascii") + b"\0")
        code_hasher.update(content)

    # Identificar el contenido de los datos usado en esta ejecución.
    data_sha256 = hashlib.sha256(
        data.to_csv(index=False).encode("utf-8")
    ).hexdigest()

    excluded_predictors = [
        column
        for column in ("Id", "SaleType", "SaleCondition")
        if column in features.columns
    ]

    predictor_columns = [
        column
        for column in features.columns
        if column not in excluded_predictors
    ]

    estimator = evaluation_model.steps[-1][1]
    requirements_path = application_dir / "requirements.txt"

    metadata: dict[str, Any] = {
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "problem_type": "regression",
        "target": target_column,
        "model": "sklearn.linear_model.LinearRegression",
        "model_parameters": estimator.get_params(deep=False),
        "seed": random_state,
        "source_data": source_data,
        "data_source": (
            "https://www.kaggle.com/c/"
            "house-prices-advanced-regression-techniques/data"
        ),
        "data_sha256": data_sha256,
        "data_fingerprint_method": (
            "SHA-256 of the loaded DataFrame serialized as CSV "
            "without its index, using UTF-8"
        ),
        "source_rows": len(data),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "split_strategy": "random_holdout",
        "shuffle": True,
        "test_size": test_size,
        "training_outlier_ids_removed": training_outlier_ids,
        "evaluation_metrics": ["mae", "rmse", "r2"],
        "input_columns": [
            str(column)
            for column in predictor_columns
        ],
        "input_dtypes": {
            str(column): str(features[column].dtype)
            for column in predictor_columns
        },
        "input_types": {
            str(column): (
                "numeric"
                if pd.api.types.is_numeric_dtype(features[column])
                else "categorical"
            )
            for column in predictor_columns
        },
        "excluded_predictors": excluded_predictors,
        "preprocessing": [
            "SemanticPreprocessor",
            "SimpleImputer(median) + StandardScaler for numeric columns",
            "SimpleImputer(most_frequent) + OneHotEncoder for categorical columns",
        ],
        "python_version": platform.python_version(),
        "dependencies_file": "requirements.txt",
        "dependencies_sha256": hashlib.sha256(
            requirements_path.read_bytes()
        ).hexdigest(),
        "library_versions": {
            "numpy": package_version("numpy"),
            "pandas": pd.__version__,
            "scikit-learn": package_version("scikit-learn"),
            "scipy": package_version("scipy"),
            "joblib": package_version("joblib"),
        },
        "code_reference": {
            "method": "sha256",
            "value": code_hasher.hexdigest(),
            "files": [
                path.relative_to(application_dir).as_posix()
                for path in code_files
            ],
        },
    }

    model_path, metrics_path, metadata_path = save_training_artifacts(
        evaluation_model,
        output_dir,
        version,
        metrics,
        metadata,
    )

    return TrainingArtifacts(
        model_path,
        metrics_path,
        metadata_path,
        metrics,
    )