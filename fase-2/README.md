# Fase 2 — Scripts, versionamiento y Docker

Proyecto Integrador — Modelos y Simulación de Sistemas I
Universidad de Antioquia · 2026-II

## Integrantes

- Nicole Mariana Adarve Tangarife
- Isabella Sánchez Mejía
- María Alejandra Otálvaro Ramírez

## 1. Descripción del proyecto y continuidad con la Fase 1

El proyecto estima el precio de venta de viviendas residenciales de Ames, Iowa, mediante aprendizaje supervisado de **regresión**. La variable objetivo es `SalePrice` y las variables de entrada describen características físicas, estructurales, de ubicación y calidad de las viviendas. Los datos proceden de la competición [House Prices — Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data), cuyo conjunto de entrenamiento contiene **1.460 viviendas**.

En la **Fase 1** se desarrolló el análisis y el entrenamiento de un modelo de regresión lineal (`LinearRegression`) en `fase-1/notebook.ipynb`, con preprocesamiento y comparación frente a un modelo base (`DummyRegressor`, estrategia `mean`).

En la **Fase 2** se mantiene el problema y el enfoque de modelado, pero se convierte el trabajo del notebook en una aplicación reutilizable. El entrenamiento y la predicción se ejecutan desde la terminal, sin abrir el notebook, mediante un pipeline persistido, modelos versionados, pruebas con `pytest` y contenedores Docker.

### Cambios y correcciones conservados respecto a la Fase 1

- Separación de los comandos `train.py` y `predict.py` y de la lógica reutilizable en el paquete `house_prices/`.
- Partición de entrenamiento/prueba **antes** de ajustar imputación, escalamiento y codificación, para evitar fuga de información.
- Tratamiento semántico de valores faltantes, distinguiendo ausencia real de dato desconocido. Los ocho nulos documentados de `MasVnrType` y `MasVnrArea` se conservan como desconocidos, no se convierten automáticamente en ausencia de mampostería.
- Uso de la variable derivada `HasGarage` y tratamiento de `GarageYrBlt` dentro del preprocesamiento.
- Exclusión de `SaleType` y `SaleCondition` como predictores, porque corresponden a información de la transacción que no necesariamente está disponible antes de la venta. `SaleCondition` se conserva únicamente para el filtro de valores atípicos documentado.
- Retiro de las viviendas con `Id` 524 y 1299, cuando cumplen las condiciones documentadas, **solo del conjunto de entrenamiento**.
- Evaluación del modelo y del `DummyRegressor` sobre la misma partición, almacenamiento de métricas y metadatos, y pruebas automatizadas.

## 2. Organización del repositorio

```text
house-prices-modelos-simulacion/
├── fase-1/
│   ├── README.md
│   ├── notebook.ipynb
│   ├── train.csv
│   ├── test.csv
│   └── modelo.joblib
├── fase-2/
│   ├── train.py
│   ├── predict.py
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── README.md
│   ├── house_prices/
│   │   ├── data.py
│   │   ├── preprocessing.py
│   │   ├── pipeline.py
│   │   ├── model.py
│   │   ├── storage.py
│   │   └── prediction.py
│   ├── data/
│   │   └── example_prediction.csv
│   ├── tests/
│   │   ├── test_pipeline.py
│   │   └── test_requisitos_fase2.py
│   ├── models/
│   │   ├── entrega-v1/
│   │   │   ├── model.joblib
│   │   │   ├── metrics.json
│   │   │   └── metadata.json
│   │   └── entrega-v2/
│   │       ├── model.joblib
│   │       ├── metrics.json
│   │       └── metadata.json
│   └── outputs/              # Se genera al guardar predicciones
├── pytest.ini
└── .gitignore
```

Los scripts son interfaces de línea de comandos con `argparse`. Las responsabilidades se distribuyen así:

| Archivo | Responsabilidad |
|---|---|
| `data.py` | Lectura y validación de datos y columnas |
| `preprocessing.py` | Transformaciones semánticas de las variables |
| `pipeline.py` | Construcción del pipeline de regresión |
| `model.py` | Partición, entrenamiento y evaluación |
| `storage.py` | Guardado y carga de versiones y artefactos |
| `prediction.py` | Validación, predicción y escritura del CSV de salida |

Los directorios `models/` pueden contener versiones adicionales generadas para comprobaciones locales. Esas versiones no sustituyen a `entrega-v1` y `entrega-v2`.

## 3. Requisitos y preparación local

El proyecto fue ejecutado en **Windows con Python 3.13.14**. La imagen Docker utiliza **Python 3.13** (`python:3.13-slim`).

Las dependencias de `fase-2/requirements.txt` están fijadas:

| Biblioteca | Versión |
|---|---|
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| joblib | 1.6.0 |
| pytest | 8.4.2 |

**Todos los comandos de esta sección se ejecutan en PowerShell desde la raíz del repositorio**, salvo que se indique otra ubicación.

### 3.1. Crear el entorno virtual

```powershell
python -m venv fase-2/.venv
```

También puede utilizarse `py -3.13 -m venv fase-2/.venv` en equipos con el lanzador `py` instalado.

Instalar las dependencias utilizando el Python del entorno:

```powershell
.\fase-2\.venv\Scripts\python.exe -m pip install -r fase-2/requirements.txt
```

Opcionalmente, activar el entorno:

```powershell
.\fase-2\.venv\Scripts\Activate.ps1
```

Si PowerShell restringe la activación, ejecutar los scripts con `.\fase-2\.venv\Scripts\python.exe` en lugar de activar el entorno.

Comprobar la instalación:

```powershell
.\fase-2\.venv\Scripts\python.exe --version
.\fase-2\.venv\Scripts\python.exe -m pip check
```

En los siguientes ejemplos locales, `python` hace referencia al intérprete del entorno virtual activado.

### 3.2. Consultar la ayuda de los comandos

```powershell
python fase-2/train.py --help
python fase-2/predict.py --help
```

## 4. Datos de entrada y esquema de predicción

**Fuente:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

| Archivo | Uso |
|---|---|
| `fase-1/train.csv` | Entrenamiento y evaluación; incluye `SalePrice` |
| `fase-1/test.csv` | Predicción sobre viviendas nuevas; no incluye `SalePrice` |
| `fase-2/data/example_prediction.csv` | Ejemplo pequeño con el esquema real para ejecutar `predict.py` |

El repositorio incluye estos archivos, por lo que no se necesita una sesión abierta de Kaggle ni ejecutar primero el notebook de Fase 1.

**Esquema del CSV de predicción:** debe contener las columnas predictoras esperadas por la versión del modelo guardado, con tipos compatibles. **No debe incluir `SalePrice`.** `GarageType` es una columna comprobada explícitamente durante la validación. `Id`, `SaleType` y `SaleCondition` son opcionales al predecir porque no son predictores del modelo; si aparece `Id`, se conserva en la salida.

- Los valores faltantes permitidos se tratan con el pipeline previamente ajustado; una celda nula no equivale a una columna obligatoria ausente.
- Las columnas se reordenan según el esquema guardado; las adicionales se ignoran.
- Las categorías nuevas se admiten mediante la codificación One-Hot configurada.
- Si falta una columna obligatoria, hay tipos incompatibles o no existe el modelo, se informa el error y no se genera una salida aparentemente válida.

**Salida:** un CSV con `SalePricePrediction` y, cuando se incluye en la entrada, `Id`, manteniendo el orden y la cantidad de filas.

Ejemplo de encabezado y resultado:

```csv
Id,SalePricePrediction
1461,113650.17525962926
```

## 5. Entrenamiento con `train.py`

Desde la **raíz del repositorio**, con el entorno activado:

```powershell
python fase-2/train.py --data fase-1/train.csv --target SalePrice --output-dir fase-2/models --version nueva-v1 --seed 42 --test-size 0.2
```

Los argumentos son:

| Argumento | Función |
|---|---|
| `--data` | Ruta del CSV etiquetado |
| `--target` | Columna objetivo (`SalePrice`) |
| `--output-dir` | Carpeta donde guardar versiones |
| `--version` | Identificador único para esta ejecución |
| `--seed` | Semilla para reproducir la partición (por defecto, 42) |
| `--test-size` | Proporción de evaluación (por defecto, 0.2) |

Se realiza una partición aleatoria **80/20**. El preprocesamiento se ajusta exclusivamente con los datos de entrenamiento y se aplica al conjunto de prueba sin reajustarlo. El comando también compara la regresión lineal con `DummyRegressor`.

La ejecución crea, por ejemplo:

```text
fase-2/models/nueva-v1/
  model.joblib
  metrics.json
  metadata.json
```

Si el identificador ya existe, el entrenamiento rechaza la sobrescritura. Debe usarse una versión nueva.

## 6. Predicción con `predict.py`

Para utilizar una versión existente sin volver a entrenar ni ajustar transformaciones, ejecutar desde la **raíz del repositorio**:

```powershell
python fase-2/predict.py --model fase-2/models/entrega-v1 --input fase-2/data/example_prediction.csv --output fase-2/outputs/predicciones_ejemplo.csv
```

`--model` acepta la carpeta de una versión o su archivo `model.joblib`; `--input` (alias `--data`) indica el CSV sin `SalePrice`; `--output` indica dónde guardar los resultados.

Para generar predicciones sobre `fase-1/test.csv`:

```powershell
python fase-2/predict.py --model fase-2/models/entrega-v1 --input fase-1/test.csv --output fase-2/outputs/predicciones_test.csv
```

La carpeta de salida se crea cuando es necesario. El CSV mantiene la correspondencia de las predicciones con los registros de entrada.

## 7. Preprocesamiento, evaluación y reproducibilidad

El artefacto `model.joblib` contiene el **pipeline completo**, no únicamente el estimador:

- `SemanticPreprocessor`: aplica las reglas de ausencias, construye `HasGarage` y prepara variables como `GarageYrBlt`.
- Variables numéricas: imputación con mediana y `StandardScaler`.
- Variables categóricas: imputación con categoría más frecuente y `OneHotEncoder`.
- Modelo final: `LinearRegression`.

El conjunto de evaluación queda separado antes de ajustar las transformaciones. `SalePrice` no se usa como predictor. `SaleType` y `SaleCondition` se excluyen del modelo principal por el momento en que se busca estimar el precio. `YrSold` y `MoSold` se conservan como fecha prevista.

Los valores atípicos documentados (`Id` 524 y 1299 con `SaleCondition == "Partial"` y `GrLivArea > 4000`) se retiran únicamente del entrenamiento, no del conjunto de prueba.

### Resultados obtenidos

| Métrica | Regresión lineal | DummyRegressor |
|---|---:|---:|
| MAE | 21.182,45 | 62.580,40 |
| RMSE | 64.975,84 | 87.619,50 |
| R² | 0,4496 | -0,0009 |

La regresión lineal supera al modelo base en las tres métricas reportadas. Las cifras corresponden al conjunto de evaluación con la partición y configuración documentadas; no representan una garantía de precisión para cualquier vivienda.

**Limitaciones conocidas:** un R² cercano a 0,45 deja una parte importante de la variabilidad sin explicar; el resultado depende de los datos disponibles y del escenario de Ames. La imputación y la codificación siguen las reglas aprendidas durante el entrenamiento. Entre equipos o bibliotecas de álgebra lineal pueden existir pequeñas diferencias numéricas, aunque se conserve la semilla 42.

En una comprobación previa documentada, dos entrenamientos con los mismos datos y configuración (`v1` y `comprobacion_punto8`) presentaron las mismas métricas y sus predicciones se compararon con `numpy.testing.assert_allclose` (`rtol=1e-10`, `atol=1e-8`).

## 8. Versionamiento y persistencia de modelos

Las dos versiones destinadas a la entrega son:

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

- **`model.joblib`**: pipeline y estimador ajustados.
- **`metrics.json`**: métricas del modelo y del `DummyRegressor`, partición y parámetros de evaluación.
- **`metadata.json`**: versión, fecha UTC, variable objetivo, algoritmo y parámetros, columnas y tipos, referencia al dataset, cantidades de registros, semilla, partición, versiones de bibliotecas y huellas SHA-256 de datos, código y dependencias.

Las versiones `entrega-v1` y `entrega-v2` se generaron con la misma semilla, datos y configuración, por lo que reportan las mismas métricas. Ambas pueden cargarse para predecir de manera independiente:

```powershell
python fase-2/predict.py --model fase-2/models/entrega-v1 --input fase-2/data/example_prediction.csv --output fase-2/outputs/entrega-v1.csv
python fase-2/predict.py --model fase-2/models/entrega-v2 --input fase-2/data/example_prediction.csv --output fase-2/outputs/entrega-v2.csv
```

Las versiones de entrega deben estar disponibles en el repositorio o mediante un enlace de descarga documentado, junto con sus tres artefactos. Las versiones temporales de práctica o validación no deben confundirse con ellas.

## 9. Pruebas automatizadas con `pytest`

Las pruebas se encuentran en:

- `fase-2/tests/test_pipeline.py`: preprocesamiento, entrenamiento, artefactos, validación, predicciones y ayuda de los comandos.
- `fase-2/tests/test_requisitos_fase2.py`: tres comprobaciones complementarias de valores faltantes, versionamiento y persistencia.

Desde la **raíz del repositorio**, con el entorno activado:

```powershell
python -m pytest fase-2/tests -v
```

O desde **`fase-2`**, ejecutando el Python local del entorno:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Para ejecutar solo las pruebas complementarias desde `fase-2`:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_requisitos_fase2.py -v
```

### Comportamientos comprobados

| Comportamiento | Comprobación |
|---|---|
| Entrenamiento y almacenamiento | Generación y carga de modelo, métricas y metadatos |
| Carga y predicción | Predicción por fila y mantenimiento del orden |
| Valores faltantes | Predicciones válidas con nulos permitidos |
| Entrada inválida | Errores ante columnas faltantes o incompatibles |
| Versionamiento | Creación de versiones independientes y rechazo de sobrescrituras |
| Persistencia | Equivalencia numérica antes y después de serializar con `joblib` |
| Interfaz CLI | Disponibilidad de ayuda y validación de argumentos |

Las pruebas complementarias utilizan datos sintéticos y carpetas temporales de `pytest` (`tmp_path`) para no alterar las versiones de entrega. La equivalencia de predicciones se comprueba con `numpy.testing.assert_allclose` (`rtol=1e-6`, `atol=1e-6`).

**Resultado verificado en Windows:** 13 pruebas aprobadas, sin fallos (10 originales y 3 complementarias), con Python 3.13.14 y pytest 8.4.2.

## 10. Ejecución con Docker

Docker Desktop debe estar iniciado y utilizando contenedores Linux. El `Dockerfile` usa `python:3.13-slim`, instala `requirements.txt` y copia los scripts y el paquete reutilizable. El entrenamiento **no** se ejecuta durante `docker build`.

El `Dockerfile` configura `ENTRYPOINT ["python"]`; por tanto, en `docker run` se indica directamente `train.py` o `predict.py`, **sin escribir `python` nuevamente**.

**Todos los comandos Docker siguientes se ejecutan desde la carpeta `fase-2` en PowerShell.**

### 10.1. Construir la imagen

```powershell
docker build -t house-prices-fase2 .
```

### 10.2. Comprobar la ayuda

```powershell
docker run --rm house-prices-fase2 train.py --help
docker run --rm house-prices-fase2 predict.py --help
```

### 10.3. Entrenar con un montaje persistente

```powershell
docker run --rm `
  -v "${PWD}/../fase-1/train.csv:/app/data/train.csv:ro" `
  -v "${PWD}/models:/app/models" `
  house-prices-fase2 train.py `
  --data /app/data/train.csv `
  --target SalePrice `
  --output-dir /app/models `
  --version docker-validacion-v1 `
  --seed 42 `
  --test-size 0.2
```

El montaje de datos es de solo lectura (`:ro`). El modelo, las métricas y los metadatos quedan en `fase-2/models/docker-validacion-v1/` **en el computador anfitrión**, aunque el contenedor se elimine con `--rm`.

Si esa versión ya existe, hay que elegir otro identificador; no se sobrescriben sus artefactos.

### 10.4. Generar predicciones con el modelo guardado

```powershell
docker run --rm `
  -v "${PWD}/models:/app/models:ro" `
  -v "${PWD}/../fase-1/test.csv:/app/data/test.csv:ro" `
  -v "${PWD}/outputs:/app/outputs" `
  house-prices-fase2 predict.py `
  --model /app/models/docker-validacion-v1 `
  --input /app/data/test.csv `
  --output /app/outputs/predicciones_docker.csv
```

Las predicciones permanecen en `fase-2/outputs/predicciones_docker.csv`. Para inspeccionar las primeras filas desde PowerShell, aún dentro de `fase-2`:

```powershell
Get-Content .\outputs\predicciones_docker.csv -TotalCount 6
```

Para usar el archivo de ejemplo en lugar de `test.csv`, se puede montar `data/example_prediction.csv` como entrada, adaptando la ruta de `--input`.

### 10.5. Ejecutar las pruebas con Docker

Las pruebas no se copian dentro de la imagen de producción debido a `.dockerignore`; se montan temporalmente los archivos necesarios.

Desde **`fase-2`**, en PowerShell:

```powershell
docker run --rm --entrypoint python `
  -v "${PWD}/tests:/fase-2/tests:ro" `
  -v "${PWD}/train.py:/fase-2/train.py:ro" `
  -v "${PWD}/predict.py:/fase-2/predict.py:ro" `
  -v "${PWD}/house_prices:/fase-2/house_prices:ro" `
  -v "${PWD}/requirements.txt:/fase-2/requirements.txt:ro" `
  house-prices-fase2 -m pytest /fase-2/tests -v -p no:cacheprovider
```

`-p no:cacheprovider` desactiva la caché de `pytest` para evitar la advertencia de escritura en la carpeta de pruebas montada como solo lectura.

### 10.6. Evidencia de ejecución

Se verificaron las siguientes operaciones dentro de Docker:

| Validación | Resultado |
|---|---|
| Construcción de la imagen | Correcta |
| Ayuda de `train.py` y `predict.py` | Disponible |
| Pruebas automatizadas | **13 aprobadas, 0 fallidas** |
| Entrenamiento desde `fase-1/train.csv` | Correcto |
| Persistencia de `model.joblib`, `metrics.json`, `metadata.json` | Correcta en carpeta del equipo |
| Predicción con `fase-1/test.csv` | Correcta |
| CSV `outputs/predicciones_docker.csv` | Generado y consultado correctamente |

La ejecución de entrenamiento en Docker produjo MAE = **21.182,45**, RMSE = **64.975,84** y R² = **0,4496** para regresión lineal; para `DummyRegressor`, MAE = **62.580,40**, RMSE = **87.619,50** y R² = **-0,0009**.

La ejecución original de las pruebas en Docker mostró **13 aprobadas** y una advertencia sobre la caché en un directorio de solo lectura; la opción `-p no:cacheprovider` evita esa advertencia.

## 11. Limitaciones y consideraciones de uso

- El modelo fue evaluado con datos de Ames, Iowa; su rendimiento puede cambiar con viviendas de otras zonas o periodos.
- Los resultados no reemplazan una valoración inmobiliaria profesional.
- `R² = 0,4496` muestra margen de mejora; no se busca en esta fase cambiar obligatoriamente el algoritmo ni mejorar la métrica.
- Para predecir se requiere el esquema de variables y los tipos esperados por el modelo guardado.
- El archivo `model.joblib` debe cargarse únicamente desde fuentes confiables, dado que los artefactos serializados con `joblib` no deben considerarse seguros cuando provienen de terceros desconocidos.
- No se exige una API REST, monitoreo, MLflow, DVC, Docker Compose ni publicar la imagen Docker en la nube en esta fase.

## 12. Fuentes y referencias

- **Datos del proyecto:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).
- **Antecedentes y decisiones técnicas:** [`fase-1/README.md`](../fase-1/README.md) y `fase-1/notebook.ipynb`.
- **Especificaciones del trabajo:** *Especificación Técnica — Fase 2: Scripts y Docker*, Proyecto Integrador 2026-II, Modelos y Simulación de Sistemas I.
- **Bibliotecas utilizadas:** `scikit-learn`, `pandas`, `numpy`, `joblib` y `pytest`, en las versiones fijadas en `requirements.txt`.
