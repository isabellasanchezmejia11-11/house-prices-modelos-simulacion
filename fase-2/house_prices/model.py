"""Construcción, entrenamiento y persistencia del modelo de regresión."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from house_prices.preprocessing import SemanticPreprocessor


def build_model() -> Pipeline:
    """Crea el pipeline completo; los imputadores aprenden solo durante fit."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    column_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                make_column_selector(dtype_include="number"),
            ),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_include=["object", "string"]),
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("semantic_preprocessing", SemanticPreprocessor()),
            ("preprocessing", column_preprocessor),
            ("regression", LinearRegression()),
        ]
    )


def _next_model_path(models_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = models_dir / f"house_prices_{timestamp}.joblib"
    suffix = 1
    while path.exists():
        path = models_dir / f"house_prices_{timestamp}_{suffix}.joblib"
        suffix += 1
    return path


def train_model(
    data: pd.DataFrame,
    models_dir: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Path, dict[str, float]]:
    """Evalúa un split holdout y guarda el pipeline ajustado solo con train."""
    if "SalePrice" not in data.columns:
        raise ValueError("El CSV de entrenamiento debe incluir la columna 'SalePrice'.")
    if len(data) < 2:
        raise ValueError("Se requieren al menos dos filas para entrenar y evaluar.")

    target = pd.to_numeric(data["SalePrice"], errors="raise")
    if target.isna().any():
        raise ValueError("La columna 'SalePrice' no puede contener valores faltantes.")
    features = data.drop(columns=["SalePrice"])

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
    outlier_mask = (
        X_train["Id"].isin([524, 1299])
        & X_train["SaleCondition"].eq("Partial")
        & pd.to_numeric(X_train["GrLivArea"], errors="coerce").gt(4000)
    )
    X_train = X_train.loc[~outlier_mask].copy()
    y_train = y_train.loc[~outlier_mask]

    evaluation_model = build_model()
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }

    output_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = _next_model_path(output_dir)
    joblib.dump(evaluation_model, model_path)
    return model_path, metrics
