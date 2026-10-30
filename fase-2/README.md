# Fase 2 — Scripts y Docker

Esta fase conserva la regresión lineal y los datos de Ames usados en Fase 1,
pero permite entrenar y predecir desde la línea de comandos. Las transformaciones
semánticas y el preprocesamiento estadístico forman parte de un único pipeline
persistido con el modelo.

## Organización y comandos

Los puntos de entrada interpretan argumentos y coordinan módulos reutilizables:

- `train.py` y `predict.py`: interfaces CLI con `argparse`; no leen datos ni
  entrenan al importarse.
- `house_prices/data.py`: lectura de CSV y validación de columnas, filas y
  variable objetivo.
- `house_prices/pipeline.py` y `preprocessing.py`: construcción del pipeline y
  transformaciones compartidas por ajuste y predicción.
- `house_prices/model.py`: partición, ajuste y evaluación.
- `house_prices/storage.py`: guardado versionado y carga de modelos.
- `house_prices/prediction.py`: predicción validada y escritura del CSV de salida.

## Preparación local

El entorno local se verificó en Windows con Python 3.13.14.
Las dependencias están fijadas en `requirements.txt`, incluido
`pytest==8.4.2`.

Los siguientes comandos se ejecutan desde la raíz del repositorio,
usando PowerShell.

### Crear el entorno e instalar las dependencias

```powershell
py -3.13 -m venv fase-2/.venv
.\fase-2\.venv\Scripts\python.exe -m pip install -r fase-2/requirements.txt
```

Para ejecutar los comandos locales de este README, activa el entorno:

```powershell
.\fase-2\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activación, puedes usar directamente
`.\fase-2\.venv\Scripts\python.exe` en lugar de `python`.

Comprueba la versión de Python y la compatibilidad de las dependencias:

```powershell
python --version
python -m pip check
```

### Consultar la ayuda

```powershell
python fase-2/train.py --help
python fase-2/predict.py --help
```

### Entrenar

El archivo `fase-1/train.csv` debe estar disponible; se incluye en este
repositorio. Para evaluar el modelo con una partición reproducible 80/20 y
guardar una nueva versión:

```powershell
python fase-2/train.py --data fase-1/train.csv --target SalePrice --output-dir fase-2/models --version v1 --seed 42 --test-size 0.2
```

Si `v1` ya existe, utiliza un identificador nuevo. Una versión existente
no se sobrescribe.

`--target` debe nombrar una columna numérica del CSV; para Ames es
`SalePrice`. `--seed` y `--test-size` controlan la partición aleatoria
reproducible.

El comando imprime las métricas de regresión lineal y `DummyRegressor`
medidas sobre el mismo conjunto de prueba, además de las rutas generadas:

```text
fase-2/models/v1/model.joblib
fase-2/models/v1/metrics.json
fase-2/models/v1/metadata.json
```

La partición se realiza antes del preprocesamiento. Los outliers documentados
(Id 524 y 1299, con `SaleCondition == "Partial"` y `GrLivArea > 4000`) se
excluyen solo del entrenamiento.

El pipeline guardado incorpora las transformaciones ajustadas exclusivamente
con ese conjunto de entrenamiento filtrado; el conjunto de prueba no participa
en el ajuste.

### Predecir

Se puede indicar el archivo `model.joblib` o directamente el directorio
de la versión.

El repositorio incluye `fase-2/data/example_prediction.csv`, un ejemplo pequeño
con el esquema real del proyecto y sin `SalePrice`.

```powershell
python fase-2/predict.py --model fase-2/models/v1 --input fase-2/data/example_prediction.csv --output fase-2/outputs/predictions.csv
```

La entrada debe incluir todas las columnas predictoras usadas en el
entrenamiento. `Id`, `SaleType` y `SaleCondition` son opcionales porque no se
usan como predictores; si está presente, `Id` se conserva en la salida.

Las columnas adicionales se ignoran y las columnas se reordenan según el
esquema del pipeline guardado. Las categorías no observadas durante el
entrenamiento se aceptan y One-Hot las ignora.

Los valores nulos se procesan con las transformaciones guardadas. Una columna
obligatoria ausente o un valor de tipo incompatible produce un error
comprensible y no genera resultados nuevos.

La salida contiene `SalePricePrediction` y, si estaba disponible, `Id`.
El orden de las filas corresponde al de la entrada.

## Acceso a los datos

La fuente es la competición
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

El repositorio incluye los archivos necesarios:

- `fase-1/train.csv`: datos etiquetados para entrenar y evaluar;
  contiene la variable objetivo `SalePrice`.
- `fase-1/test.csv`: datos de Kaggle sin la variable objetivo para predecir.
- `fase-2/data/example_prediction.csv`: ejemplo pequeño de predicción
  sin `SalePrice`.

Las rutas anteriores son relativas a la raíz del repositorio.

Como los datos están incluidos, no es necesario descargar archivos,
configurar credenciales ni mantener una sesión de Kaggle o Colab activa.
Tampoco es necesario ejecutar el notebook de la fase 1.

El `.gitignore` excluye entornos virtuales, cachés, archivos temporales,
credenciales y predicciones de comprobación. Los datos incluidos y las
versiones del modelo se conservan para permitir la reproducción del proyecto.

## Versionamiento del modelo

Se generaron con `train.py` las versiones `entrega-v1` y `entrega-v2`.
Ambas utilizan semilla 42, partición de prueba del 20 % y regresión lineal.

Cada versión se conserva en su propia carpeta:

```text
fase-2/models/
  entrega-v1/
    model.joblib
    metrics.json
    metadata.json
  entrega-v2/
    model.joblib
    metrics.json
    metadata.json
```

`model.joblib` contiene el modelo y su preprocesamiento ajustado.
`metrics.json` conserva las métricas del modelo y del modelo base.
`metadata.json` registra la versión, fecha, algoritmo, parámetros,
variables y tipos de entrada, datos utilizados, partición, semilla,
métricas de evaluación, versión de Python y dependencias.

La referencia al código se registra mediante una huella SHA-256 de los
archivos Python de la aplicación. También se guardan huellas de los datos
procesados y de `requirements.txt`.

Los identificadores existentes se rechazan para evitar sobrescrituras.
La prueba automatizada de rechazo de versiones repetidas pasó.

Se comprobó la carga y predicción de ambas versiones. Desde la raíz
del repositorio, con el entorno virtual activo:

```powershell
python fase-2/predict.py --model fase-2/models/entrega-v1 --input fase-2/data/example_prediction.csv --output fase-2/outputs/entrega-v1.csv
python fase-2/predict.py --model fase-2/models/entrega-v2 --input fase-2/data/example_prediction.csv --output fase-2/outputs/entrega-v2.csv
```

Ambas ejecuciones finalizaron correctamente. Las dos versiones tienen
las mismas métricas porque utilizan los mismos datos y configuración.

Las carpetas de ambas versiones deben incluirse en la entrega del
repositorio, junto con sus modelos, métricas y metadatos.

## Ejecución con Docker

Se requiere Docker Desktop iniciado y configurado para contenedores Linux.

La imagen utiliza `python:3.13-slim`, compatible con Python 3.13 del
entorno local. Instala las dependencias de `requirements.txt` y permite
ejecutar entrenamiento y predicción desde la misma imagen.

El entrenamiento no se ejecuta durante la construcción de la imagen.
El archivo `.dockerignore` excluye entornos virtuales, cachés,
credenciales, datos y resultados del contexto de construcción.

Los siguientes comandos se ejecutan desde la raíz del repositorio,
en PowerShell.

### Construir la imagen

```powershell
docker build -t house-prices-fase2 -f fase-2/Dockerfile fase-2
```

El Dockerfile utiliza `ENTRYPOINT ["python"]`, por lo que los comandos
indican directamente `train.py` o `predict.py`.

### Entrenar y guardar los artefactos

```powershell
docker run --rm -v "${PWD}\fase-1:/data:ro" -v "${PWD}\fase-2\models:/app/models" house-prices-fase2 train.py --data /data/train.csv --target SalePrice --output-dir /app/models --version entrega-docker-v1 --seed 42 --test-size 0.2
```

Los datos se montan en modo lectura. Los artefactos se guardan en
`fase-2/models/entrega-docker-v1/` del equipo anfitrión:

- `model.joblib`
- `metrics.json`
- `metadata.json`

Si esa versión ya existe, debe utilizarse un identificador nuevo.

### Generar predicciones

```powershell
docker run --rm -v "${PWD}\fase-2\data:/app/data:ro" -v "${PWD}\fase-2\models:/app/models:ro" -v "${PWD}\fase-2\outputs:/app/outputs" house-prices-fase2 predict.py --model /app/models/entrega-docker-v1 --input /app/data/example_prediction.csv --output /app/outputs/predicciones-docker.csv
```

Los datos y modelos se montan en modo lectura. Las predicciones se
guardan en `fase-2/outputs/predicciones-docker.csv` del equipo anfitrión.

### Evidencia de ejecución y persistencia

El 28 de octubre de 2026 se verificaron la construcción de la imagen,
el entrenamiento y la predicción mediante Docker Desktop con
contenedores Linux.

El entrenamiento mostró:

- Regresión lineal: MAE=21182.45, RMSE=64975.84 y R²=0.4496.
- DummyRegressor: MAE=62580.40, RMSE=87619.50 y R²=-0.0009.

La predicción sobre el ejemplo produjo:

```text
Id,SalePricePrediction
1461,113650.17525962926
```

Después de finalizar los comandos con `--rm`, se comprobó desde
PowerShell que los tres artefactos y el CSV permanecían en el equipo:

```powershell
Get-ChildItem .\fase-2\models\entrega-docker-v1
Get-Content .\fase-2\outputs\predicciones-docker.csv
```

Los montajes permiten conservar los resultados aunque se eliminen los
contenedores. No se requiere Docker Compose ni publicar la imagen.

## Preprocesamiento y momento de predicción

- `Id` se elimina del conjunto de predictores; se usa durante el entrenamiento
  únicamente para reproducir el filtro documentado de los dos outliers.
- Las categorías de variables de espacios ausentes, como sótano, garaje o
  piscina, se representan con la categoría `"None"` y se añade `HasGarage`
  dentro del pipeline.
- Los ocho valores realmente faltantes de `MasVnrType` y `MasVnrArea`
  se conservan como desconocidos. No se convierten en ausencia de mampostería:
  la categoría desconocida se imputa con la categoría más frecuente y el área
  numérica con la mediana. Ambas estadísticas se aprenden únicamente al
  ajustar el pipeline con los datos de entrenamiento.
- Las variables numéricas se imputan con mediana y se escalan; las categóricas
  se imputan con la categoría más frecuente y se codifican con One-Hot.
  Las categorías nuevas durante la predicción se toleran.
- La predicción se plantea antes de concretar la venta. Se conservan `YrSold`
  y `MoSold` como fecha prevista, pero `SaleType` y `SaleCondition` se excluyen
  porque describen la transacción y no están disponibles al tasar o publicar
  la vivienda.
- `SalePrice` se usa exclusivamente como objetivo durante el entrenamiento
  y se descarta si aparece en el CSV de predicción.
- La evaluación utiliza una partición aleatoria reproducible con semilla 42.
  La imputación, el escalamiento y la codificación se ajustan dentro del
  pipeline solo con los datos de entrenamiento.

## Reproducibilidad

El artefacto `model.joblib` conserva el pipeline completo: las
transformaciones semánticas, el preprocesamiento ajustado y la regresión
lineal. La predicción carga este pipeline sin volver a entrenar ni ajustar
las transformaciones.

El pipeline conserva el esquema de entrada y reordena las variables según
el entrenamiento. También reutiliza las reglas de valores faltantes y la
codificación de categorías aprendidas.

La partición se realiza antes de ajustar el preprocesamiento, con
`seed=42` y `test_size=0.2`. La imputación, el escalamiento y la
codificación se ajustan únicamente con los datos de entrenamiento.

`SalePrice` no se utiliza como predictor y se excluyen `SaleType` y
`SaleCondition` por corresponder a información de la transacción.

### Comprobación realizada

En Windows con Python 3.13.14 se ejecutaron dos entrenamientos con los
mismos datos, configuración y entorno: `v1` y `comprobacion_punto8`.

Ambos mostraron los siguientes resultados:

| Métrica | Regresión lineal | DummyRegressor |
|---|---:|---:|
| MAE | 21182.45 | 62580.40 |
| RMSE | 64975.84 | 87619.50 |
| R² | 0.4496 | -0.0009 |

Se cargaron ambos modelos y se generaron predicciones sobre
`fase-2/data/example_prediction.csv`.

Las predicciones coincidieron mediante
`numpy.testing.assert_allclose`, con `rtol=1e-10` y `atol=1e-8`.

La regresión lineal utilizada no requiere una semilla propia; la semilla
controla la separación de los datos. Pueden existir pequeñas diferencias
numéricas entre sistemas, versiones de dependencias o bibliotecas de
álgebra lineal.

La reproducibilidad se espera al conservar los mismos datos,
configuración y entorno.

## Pruebas

Con el entorno virtual activo, desde la raíz del repositorio:

```powershell
python -m pytest fase-2/tests -v
```

También se pueden ejecutar desde la carpeta `fase-2`:

```powershell
python -m pytest -v
```

`requirements-dev.txt` remite a `requirements.txt`, que ya incluye pytest.

Las pruebas cubren las reglas semánticas, la distinción entre ausencia y dato
desconocido de mampostería, la validación del objetivo configurable, la
comparación con el modelo base, la persistencia de modelo, métricas y metadatos,
el manejo de entradas inválidas y la disponibilidad de ayuda por comandos.

En la comprobación local con Windows, Python 3.13.14 y pytest 8.4.2,
las 10 pruebas existentes finalizaron correctamente.

## Pruebas complementarias del punto 12

Las pruebas complementarias se implementaron en `fase-2/tests/test_requisitos_fase2.py`, sin reemplazar las pruebas existentes de `fase-2/tests/test_pipeline.py`.

- **Predicción con valores faltantes (`test_prediccion_con_valores_faltantes`)**: verifica que el pipeline genere predicciones finitas con valores nulos permitidos y conserve la cantidad y los identificadores de las viviendas.
- **Versionamiento (`test_versionamiento_de_modelos`)**: comprueba la creación y carga de dos versiones, el rechazo de un identificador ya existente mediante `FileExistsError` y la conservación del contenido del modelo original.
- **Persistencia (`test_predicciones_se_conservan_al_guardar_modelo`)**: compara las predicciones del pipeline entrenado antes de guardarlo con las obtenidas después de recuperarlo mediante `joblib`, utilizando `numpy.testing.assert_allclose` con `rtol=1e-6` y `atol=1e-6`.

Para ejecutar únicamente estas tres pruebas desde la carpeta `fase-2`:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_requisitos_fase2.py -v
```

Para ejecutar todas las pruebas desde la misma carpeta:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

**Resultado de la ejecución local:** 13 pruebas aprobadas, sin fallos (10 existentes y 3 complementarias), en Windows con Python 3.13.14 y pytest 8.4.2. Las pruebas complementarias usan datos sintéticos y directorios temporales de pytest para no modificar las versiones de entrega.
