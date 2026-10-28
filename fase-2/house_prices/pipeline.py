"""Construcción del pipeline de preprocesamiento y regresión."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
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
