# Fase 3 · API REST

API REST desarrollada con FastAPI para consultar el estado del servicio, realizar predicciones de precios de viviendas y entrenar nuevas versiones del modelo mediante solicitudes HTTP.

La implementación reutiliza las funciones de entrenamiento, predicción, validación de datos y preprocesamiento desarrolladas en la fase 2 del proyecto House Prices.

## 1. Ejecución local

Desde la raíz del repositorio, utilizando PowerShell y Python 3.13:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r fase-3/requirements.txt
python -m uvicorn api:app --app-dir fase-3 --host 0.0.0.0 --port 8000
```

Si ya existe un entorno virtual con las dependencias instaladas, puede reutilizarse.

En Linux/macOS, crear el entorno con `python3 -m venv .venv` y activarlo mediante `source .venv/bin/activate`.

La documentación interactiva de la API está disponible en:

http://localhost:8000/docs

## 2. Operaciones implementadas

### GET /health

Permite consultar el estado de ejecución de la API.

**Respuesta esperada — HTTP 200:**

```json
{
  "estado": "ok"
}
```

Esta operación confirma que el servidor responde, aunque no exista un modelo disponible.

### POST /predict

Recibe un registro JSON con las variables predictoras de una vivienda, aplica el preprocesamiento almacenado y devuelve el precio estimado junto con la versión del modelo utilizada.

La solicitud debe contener los 77 predictores obligatorios. Las columnas `Id`, `SaleType` y `SaleCondition` son opcionales.

El archivo `fase-3/data/example_request.json` contiene un ejemplo basado en la primera fila de `fase-2/data/example_prediction.csv`.

Para ejecutar una predicción desde PowerShell:

```powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data-binary "@fase-3/data/example_request.json"
```

**Respuesta obtenida — HTTP 200:**

```json
{
  "prediccion": 113650.17525962926,
  "version": "entrega-v1"
}
```

La predicción corresponde a los datos del ejemplo y puede presentar pequeñas variaciones entre entornos.

### POST /train

Permite entrenar una nueva versión del modelo reutilizando las funciones de la fase 2.

El entrenamiento utiliza la siguiente configuración:

- Archivo de datos: `fase-1/train.csv`.
- Variable objetivo: `SalePrice`.
- Proporción de evaluación: `test_size=0.2`.
- Semilla de reproducibilidad: `random_state=42`.
- Directorio de almacenamiento: `fase-2/models`.

El endpoint no requiere un cuerpo JSON.

Para ejecutarlo:

```powershell
curl.exe -X POST http://localhost:8000/train
```

También puede ejecutarse desde Swagger mediante **POST /train → Try it out → Execute**.

**Resultado obtenido en Docker — HTTP 201:**

```json
{
  "mensaje": "Modelo entrenado correctamente",
  "version": "entrenamiento-20261107T222137632057Z",
  "modelo": "/app/fase-2/models/entrenamiento-20261107T222137632057Z/model.joblib",
  "metricas": "/app/fase-2/models/entrenamiento-20261107T222137632057Z/metrics.json",
  "metadatos": "/app/fase-2/models/entrenamiento-20261107T222137632057Z/metadata.json"
}
```

Cada entrenamiento genera un identificador basado en la fecha y hora. Los artefactos se almacenan en una carpeta independiente, conservando las versiones anteriores.

Los archivos generados son:

- `model.joblib`: modelo entrenado y pipeline de preprocesamiento.
- `metrics.json`: métricas de evaluación del entrenamiento.
- `metadata.json`: información de la versión, configuración y entrenamiento.

La ejecución de `/train` no cambia automáticamente la versión activa utilizada por `/predict`.

## 3. Validación de entradas y manejo de errores

La API comprueba los campos obligatorios y sus tipos antes de realizar una predicción.

Se rechazan las solicitudes que presenten campos obligatorios ausentes, tipos incompatibles, variables desconocidas o la variable objetivo `SalePrice`.

Los campos numéricos deben enviarse como números y las variables categóricas como texto. Los valores `null` se convierten en valores ausentes para aplicar las reglas de preprocesamiento e imputación definidas en la fase 2.

Una variable presente con valor `null` no equivale a una variable obligatoria ausente.

Los errores de validación devuelven HTTP 422 con un mensaje que identifica el problema.

Cuando el modelo configurado no está disponible o resulta incompatible, `/predict` devuelve HTTP 503 con un mensaje comprensible.

Los errores relacionados con el acceso a los archivos durante el entrenamiento se gestionan mediante respuestas HTTP apropiadas.

## 4. Selección y conservación de versiones

De forma predeterminada, la API utiliza el modelo `entrega-v1`, almacenado en:

`fase-2/models/entrega-v1/model.joblib`

Para utilizar otra versión, se debe configurar la variable de entorno `MODEL_VERSION` antes de iniciar el servidor.

Ejemplo en PowerShell:

```powershell
$env:MODEL_VERSION = "entrega-v2"
python -m uvicorn api:app --app-dir fase-3 --host 0.0.0.0 --port 8000
```

Para volver a la versión anterior:

```powershell
$env:MODEL_VERSION = "entrega-v1"
python -m uvicorn api:app --app-dir fase-3 --host 0.0.0.0 --port 8000
```

Es necesario detener el servidor antes de reiniciarlo con otra configuración.

También puede utilizarse la variable opcional `MODELS_DIR` para configurar un directorio diferente de almacenamiento de modelos. Se recomienda utilizar una ruta absoluta.

Se verificó la conservación de las versiones `entrega-v1`, `entrega-v2` y las nuevas versiones generadas mediante `/train`.

## 5. Cliente de prueba

La fase 3 incluye el archivo `fase-3/client.py`, que permite comprobar el servicio utilizando la biblioteca estándar de Python.

Con la API en ejecución, desde otra terminal ubicada en la raíz del repositorio:

```powershell
python fase-3/client.py
```

El cliente consulta el endpoint `/health` y envía el archivo `fase-3/data/example_request.json` a `/predict`.

Esto permite realizar una comprobación sencilla del funcionamiento del servicio sin utilizar Swagger.

## 6. Pruebas automatizadas

Las pruebas se implementaron con pytest y el cliente de pruebas de FastAPI.

Se encuentran en:

`fase-3/tests/test_api.py`

Para instalar las dependencias necesarias:

```powershell
python -m pip install -r fase-3/requirements-dev.txt
```

Para ejecutar las pruebas desde la raíz del repositorio:

```powershell
python -m pytest fase-3/tests/test_api.py -v
```

Se verifican los siguientes escenarios:

| Prueba | Comportamiento esperado | Resultado |
|---|---|---|
| `test_health` | `/health` devuelve HTTP 200 y estado `ok` | PASSED |
| `test_predict_valid` | Entrada válida devuelve predicción y versión | PASSED |
| `test_predict_missing_required_field` | Entrada sin campo obligatorio devuelve error de validación | PASSED |

**Resultado de la ejecución:**

```text
3 passed, 1 warning in 2.84s
```

La advertencia corresponde a una deprecación relacionada con Starlette y HTTPX y no impidió superar las pruebas.

## 7. Ejecución mediante Docker

La fase 3 incluye un Dockerfile basado en Python 3.13 que ejecuta la API utilizando FastAPI y Uvicorn.

El contenedor reutiliza los módulos de la fase 2 e incorpora los archivos necesarios para el entrenamiento y la predicción.

### Construcción de la imagen

Desde la raíz del repositorio:

```powershell
docker build -f fase-3/Dockerfile -t house-prices-api-fase3 .
```

### Ejecución del contenedor

En PowerShell:

```powershell
docker run --rm --name house-prices-api -p 8000:8000 --mount "type=bind,source=$((Resolve-Path 'fase-1').Path),target=/app/fase-1,readonly" --mount "type=bind,source=$((Resolve-Path 'fase-2/models').Path),target=/app/fase-2/models" house-prices-api-fase3
```

El contenedor utiliza el puerto `8000` y dos montajes:

- `/app/fase-1`: acceso de solo lectura a los datos originales de entrenamiento.
- `/app/fase-2/models`: acceso a los modelos almacenados, permitiendo guardar nuevas versiones de forma persistente.

La documentación interactiva se encuentra disponible en:

http://localhost:8000/docs

### Resultados de la validación en Docker

Se realizaron las siguientes comprobaciones:

| Operación | Código HTTP | Resultado |
|---|---|---|
| `GET /health` | 200 | API en ejecución |
| `POST /predict` | 200 | Predicción generada con `entrega-v1` |
| `POST /train` | 201 | Nueva versión entrenada y guardada |

Durante la validación se comprobó que Docker podía acceder al archivo de entrenamiento, cargar modelos existentes y escribir nuevas versiones en el directorio compartido.

La versión generada durante la prueba en Docker fue:

`entrenamiento-20261107T222137632057Z`

Se comprobó que el entrenamiento no sobrescribió las versiones anteriores.

### Seleccionar otra versión en Docker

Para utilizar un modelo diferente se puede agregar la variable de entorno al iniciar el contenedor:

```powershell
-e MODEL_VERSION=entrega-v2
```

Esta opción debe incorporarse antes del nombre de la imagen en el comando `docker run`.

## 8. Archivos principales

La fase 3 contiene los siguientes componentes:

```text
fase-3/
├── api.py
├── client.py
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── data/
│   └── example_request.json
└── tests/
    └── test_api.py
```

- `api.py`: implementación de la API REST y sus tres endpoints.
- `client.py`: cliente Python para consultar la API.
- `Dockerfile`: configuración del contenedor.
- `requirements.txt`: dependencias de ejecución.
- `requirements-dev.txt`: dependencias de desarrollo y pruebas.
- `example_request.json`: ejemplo de entrada para predicción.
- `test_api.py`: pruebas automatizadas.

## 9. Estado de implementación

Se completaron las tres operaciones requeridas para la fase 3: `GET /health`, `POST /predict` y `POST /train`.

La implementación incluye validación de entradas, manejo de errores, reutilización de los modelos y el preprocesamiento de la fase 2, selección y conservación de versiones, cliente Python, archivo JSON de ejemplo, pruebas automatizadas y ejecución mediante Docker.

Se verificaron las tres operaciones HTTP y se aprobaron las tres pruebas automatizadas exigidas.

La validación final de reproducibilidad desde una copia limpia del repositorio queda pendiente antes de integrar los cambios en la rama `develop`.
