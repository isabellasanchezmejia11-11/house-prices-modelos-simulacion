# Fase 3 · API REST

API de consulta del estado y predicción de precios de viviendas con FastAPI. Reutiliza el modelo y el preprocesamiento guardados en fase 2.

## Ejecución local

Desde la raíz del repositorio en PowerShell, con Python 3.13:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r fase-3/requirements.txt
python -m uvicorn api:app --app-dir fase-3 --host 0.0.0.0 --port 8000
```

En otra terminal, desde la raíz:

```powershell
python fase-3/client.py
```

El cliente consulta `/health` y envía `fase-3/data/example_request.json` a `/predict`. Usa únicamente la biblioteca estándar. La documentación de la API está en http://localhost:8000/docs.

En Linux/macOS: crear el entorno con `python3 -m venv .venv` y activarlo con `source .venv/bin/activate`; los demás comandos son iguales.

## Comportamiento implementado

- `/health`: devuelve `200 {"estado":"ok"}`. Confirma que el proceso responde, aunque falte el modelo.
- `/predict`: recibe un objeto plano con los 77 predictores requeridos, aplica el preprocesamiento guardado y devuelve el precio estimado y la versión utilizada.
- El JSON de ejemplo procede de la primera fila de `fase-2/data/example_prediction.csv`, excluyendo `Id`, `SaleType` y `SaleCondition`. Estas tres columnas son opcionales y no influyen en el resultado.
- Se rechaza con 422 la ausencia de una clave obligatoria, un tipo incompatible, campos desconocidos o la variable objetivo `SalePrice`. Los campos numéricos deben enviarse como números y las categorías como texto.
- Los valores `null` se transforman a ausencias para que el pipeline aplique las reglas semánticas e imputaciones de fase 2. Una clave presente con `null` es distinta de una clave ausente.
- Si el modelo falta o no puede utilizarse, se devuelve 503 con un mensaje comprensible.

Ejemplo de solicitud desde la raíz:

```powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data-binary "@fase-3/data/example_request.json"
```

Respuesta de referencia para esa vivienda (puede variar ligeramente entre entornos):

```json
{"prediccion": 113650.17525962926, "version": "entrega-v1"}
```

## Elegir una versión existente

De forma predeterminada se carga `fase-2/models/entrega-v1/model.joblib`. Para usar la otra versión, detener el servidor y ejecutar:

```powershell
$env:MODEL_VERSION = "entrega-v2"
python -m uvicorn api:app --app-dir fase-3 --host 0.0.0.0 --port 8000
```

Para volver: `$env:MODEL_VERSION = "entrega-v1"` y reiniciar. La variable opcional `MODELS_DIR` permite indicar otro directorio de modelos; se recomienda una ruta absoluta. Sin esa variable se usa una ruta absoluta calculada desde el repositorio.

## Estado de implementación

Disponibles: `GET /health`, `POST /predict`, validación de entradas, selección de versiones existentes y cliente de ejemplo.

Pendientes para completar la fase 3: `POST /train`, pruebas automatizadas de la API, Dockerfile, validación en Docker y documentación de entrenamiento. El `pytest.ini` de la raíz conserva la configuración de las pruebas de fase 2.
