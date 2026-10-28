"""Transformaciones de datos de vivienda reutilizables en entrenamiento e inferencia."""

from __future__ import annotations

import numpy as np
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

SALE_FEATURES = ("SaleType", "SaleCondition")


class SemanticPreprocessor(BaseEstimator, TransformerMixin):
    """Aplica reglas de dominio sin aprender estadísticas de los datos."""

    def fit(self, X: pd.DataFrame, y: object = None) -> SemanticPreprocessor:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Las variables predictoras deben ser un DataFrame de pandas.")
        if "GarageType" not in X.columns:
            raise ValueError("Falta la columna requerida 'GarageType'.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Las variables predictoras deben ser un DataFrame de pandas.")
        if "GarageType" not in X.columns:
            raise ValueError("Falta la columna requerida 'GarageType'.")

        transformed = X.copy()
        transformed = transformed.drop(
            columns=["Id", "SalePrice", *SALE_FEATURES],
            errors="ignore",
        )

        for column in ABSENCE_CATEGORIES:
            if column in transformed.columns:
                values = transformed[column]
                missing = values.isna() | values.astype("string").eq("NA").fillna(False)
                transformed[column] = values.astype(object).where(~missing, "None")

        garage_type = transformed["GarageType"]
        garage_text = garage_type.astype("string")
        no_garage = (
            garage_type.isna()
            | garage_text.eq("NA").fillna(False)
            | garage_text.eq("None").fillna(False)
        )
        transformed["HasGarage"] = (~no_garage).astype("int64")

        if "GarageYrBlt" in transformed.columns:
            transformed.loc[no_garage, "GarageYrBlt"] = 0

        return transformed
