"""Transformaciones de datos de vivienda reutilizables en entrenamiento e inferencia."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


ABSENCE_CATEGORIES = (
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
)


class SemanticPreprocessor(BaseEstimator, TransformerMixin):
    """Aplica reglas de dominio sin aprender estadísticas de los datos."""

    def fit(self, X: pd.DataFrame, y: object = None) -> SemanticPreprocessor:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Las variables predictoras deben ser un DataFrame de pandas.")
        if "GarageType" not in X.columns:
            raise ValueError("Falta la columna requerida 'GarageType'.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Las variables predictoras deben ser un DataFrame de pandas.")
        if "GarageType" not in X.columns:
            raise ValueError("Falta la columna requerida 'GarageType'.")

        transformed = X.copy()
        transformed = transformed.drop(columns=["Id", "SalePrice"], errors="ignore")

        for column in ABSENCE_CATEGORIES:
            if column in transformed.columns:
                transformed[column] = transformed[column].fillna("None")

        transformed["HasGarage"] = (
            transformed["GarageType"].astype("object").fillna("None") != "None"
        ).astype("int64")

        if "GarageYrBlt" in transformed.columns:
            transformed["GarageYrBlt"] = transformed["GarageYrBlt"].fillna(0)

        return transformed
