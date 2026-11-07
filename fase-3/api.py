
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent

# Reutilizar las funciones de la fase 2
sys.path.insert(0, str(ROOT / "fase-2"))

from house_prices.data import (
    read_training_data,
    validate_prediction_data,
)
from house_prices.model import train_model
from house_prices.prediction import (
    OPTIONAL_INPUT_COLUMNS,
    _prepare_model_input,
)
from house_prices.storage import load_model

# Registro de solicitudes de la fase 4
sys.path.insert(0, str(ROOT / "fase-4"))
from registro import guardar_solicitud

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURACION DE LA API
# ============================================================

@dataclass(frozen=True)
class Settings:
    models_dir: Path = ROOT / "fase-2" / "models"
    model_version: str = "entrega-v1"

    @classmethod
    def from_env(cls):
        return cls(
            models_dir=Path(
                os.getenv("MODELS_DIR", str(cls.models_dir))
            ),
            model_version=os.getenv(
                "MODEL_VERSION",
                cls.model_version,
            ),
        )


# ============================================================
# MODELOS DE RESPUESTA
# ============================================================

class PredictionResponse(BaseModel):
    prediccion: float
    version: str


class TrainingResponse(BaseModel):
    mensaje: str
    version: str
    modelo: str
    metricas: str
    metadatos: str


# ============================================================
# VALIDACION DE ENTRADAS
# ============================================================

def validate_json_record(model, payload: dict) -> pd.DataFrame:
    schema = [str(c) for c in model.feature_names_in_]

    missing = sorted(
        set(schema) - OPTIONAL_INPUT_COLUMNS - payload.keys()
    )

    if missing:
        raise ValueError(
            f"Faltan campos obligatorios: {', '.join(missing)}."
        )

    if "SalePrice" in payload:
        raise ValueError(
            "No envíe la variable objetivo SalePrice para predecir."
        )

    unknown = sorted(payload.keys() - set(schema))

    if unknown:
        raise ValueError(
            f"Campos desconocidos: {', '.join(unknown)}."
        )

    numeric, categorical = set(), set()

    for name, _, columns in model.named_steps[
        "preprocessing"
    ].transformers_:

        if name == "numeric":
            numeric.update(str(c) for c in columns)

        elif name == "categorical":
            categorical.update(str(c) for c in columns)

    for name, value in payload.items():

        if value is None:
            continue

        if name in numeric and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"El campo '{name}' debe ser un número finito o null."
            )

        if name in categorical and not isinstance(value, str):
            raise ValueError(
                f"El campo '{name}' debe ser texto o null."
            )

        if name in OPTIONAL_INPUT_COLUMNS:

            if name == "Id":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        "El campo 'Id' debe ser un número finito o null."
                    )

            elif not isinstance(value, str):
                raise ValueError(
                    f"El campo '{name}' debe ser texto o null."
                )

    frame = pd.DataFrame([
        {
            k: np.nan if v is None else v
            for k, v in payload.items()
        }
    ])

    validate_prediction_data(frame)

    return _prepare_model_input(model, frame)


# ============================================================
# CREACION DE LA API
# ============================================================

def create_app(settings: Settings | None = None) -> FastAPI:

    config = settings or Settings.from_env()

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*",
        config.model_version,
    ):
        raise ValueError(
            "MODEL_VERSION debe ser un identificador de versión válido."
        )

    app = FastAPI(
        title="House Prices · Fase 3 y 4",
        version="1.1.0",
        description=(
            "API REST para consultar el estado, predecir "
            "precios de viviendas, entrenar modelos y "
            "registrar solicitudes de predicción."
        ),
    )

    app.state.settings = config

    # ========================================================
    # FASE 4 - REGISTRO DE SOLICITUDES
    # ========================================================

    @app.middleware("http")
    async def registrar_predicciones(
        request: Request,
        call_next,
    ):
        # Registrar exclusivamente solicitudes a /predict
        if request.url.path != "/predict":
            return await call_next(request)

        try:
            response = await call_next(request)

        except Exception:
            logger.exception(
                "Error inesperado en /predict"
            )

            guardar_solicitud(
                estado_http=500,
            )
            raise

        version = getattr(
            request.state,
            "version_prediccion",
            None,
        )

        prediccion = getattr(
            request.state,
            "valor_prediccion",
            None,
        )

        guardar_solicitud(
            estado_http=response.status_code,
            version=version,
            prediccion=prediccion,
        )

        return response

    # ========================================================
    # GET /health
    # ========================================================

    @app.get("/health")
    def health():
        return {
            "estado": "ok"
        }

    # ========================================================
    # POST /predict
    # ========================================================

    @app.post(
        "/predict",
        response_model=PredictionResponse,
    )
    def predict(
        request: Request,
        payload: dict = Body(...),
    ):

        # Cargar el modelo seleccionado
        try:
            model = load_model(
                config.models_dir / config.model_version
            )

        except (
            OSError,
            ValueError,
            TypeError,
            AttributeError,
        ) as error:

            logger.warning(
                "No se pudo cargar el modelo: %s",
                error,
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    f"Modelo '{config.model_version}' "
                    "no disponible o incompatible. "
                    "Revise MODELS_DIR y MODEL_VERSION "
                    "y seleccione una versión guardada."
                ),
            ) from error

        # Validar los datos recibidos
        try:
            frame = validate_json_record(
                model,
                payload,
            )

        except ValueError as error:

            raise HTTPException(
                status_code=422,
                detail=str(error),
            ) from error

        # Realizar la prediccion
        try:
            result = float(
                model.predict(frame)[0]
            )

            if not math.isfinite(result):
                raise ValueError(
                    "Predicción no finita."
                )

        except Exception as error:

            logger.exception(
                "Error al ejecutar el pipeline"
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "No se pudo ejecutar "
                    "el modelo configurado."
                ),
            ) from error

        # Guardar los datos de la prediccion
        # para que el middleware pueda registrarlos
        request.state.version_prediccion = (
            config.model_version
        )

        request.state.valor_prediccion = result

        return {
            "prediccion": result,
            "version": config.model_version,
        }

    # ========================================================
    # POST /train
    # ========================================================

    @app.post(
        "/train",
        response_model=TrainingResponse,
        status_code=201,
    )
    def train():

        data_path = ROOT / "fase-1" / "train.csv"

        version = (
            "entrenamiento-"
            + datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S%fZ"
            )
        )

        try:
            data = read_training_data(
                data_path,
                "SalePrice",
            )

            artifacts = train_model(
                data,
                config.models_dir,
                version=version,
                target_column="SalePrice",
                source_data=str(data_path),
                test_size=0.2,
                random_state=42,
            )

        except (
            ValueError,
            TypeError,
        ) as error:

            raise HTTPException(
                status_code=422,
                detail=str(error),
            ) from error

        except OSError as error:

            logger.exception(
                "Error al acceder a los datos o modelos"
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "No se pudo acceder a los datos "
                    "o guardar el modelo."
                ),
            ) from error

        return {
            "mensaje": "Modelo entrenado correctamente",
            "version": version,
            "modelo": str(artifacts.model_path),
            "metricas": str(artifacts.metrics_path),
            "metadatos": str(artifacts.metadata_path),
        }

    return app


app = create_app()
