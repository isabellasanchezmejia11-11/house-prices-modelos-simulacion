# Fase 2 — Scripts, versionamiento y Docker

Proyecto Integrador — Modelos y Simulación de Sistemas I  
Universidad de Antioquia · 2026-II

## Integrantes

- Nicole Mariana Adarve Tangarife
- Isabella Sánchez Mejía
- María Alejandra Otálvaro Ramírez

## 1. Descripción del proyecto y continuidad con la Fase 1

El proyecto estima el precio de venta de viviendas residenciales de Ames, Iowa, mediante aprendizaje supervisado de **regresión**. La variable objetivo es `SalePrice` y las variables de entrada describen características físicas, estructurales, de ubicación y calidad de las viviendas.

Los datos proceden de la competición [House Prices — Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data), cuyo conjunto de entrenamiento contiene **1.460 viviendas**.

En la **Fase 1** se desarrolló el análisis y el entrenamiento de un modelo de regresión lineal (`LinearRegression`) en `fase-1/notebook.ipynb`, con preprocesamiento y comparación frente a un modelo base (`DummyRegressor`, estrategia `mean`).

En la **Fase 2** se mantiene el problema y el enfoque de modelado, pero se convierte el trabajo del notebook en una aplicación reutilizable. El entrenamiento y la predicción se ejecutan desde la terminal, sin abrir el notebook, mediante un pipeline persistido, modelos versionados, pruebas con `pytest` y contenedores Docker.

### Cambios y correcciones respecto a la Fase 1

- Separación de los comandos `train.py` y `predict.py` y de la lógica reutilizable en el paquete `house_prices/`.
- Partición de entrenamiento y prueba **antes** de ajustar imputación, escalamiento y codificación, para evitar fuga de información.
- Tratamiento semántico de valores faltantes, distinguiendo ausencia real de dato desconocido. Los ocho nulos documentados de `MasVnrType` y `MasVnrArea` se conservan como desconocidos; no se convierten automáticamente en ausencia de mampostería.
- Uso de la variable derivada `HasGarage` y tratamiento de `GarageYrBlt` dentro del preprocesamiento.
- Exclusión de `SaleType` y `SaleCondition` como predictores, porque corresponden a información de la transacción que no necesariamente está disponible antes de la venta. `SaleCondition` se conserva únicamente para el filtro de valores atípicos documentado.
- Retiro de las viviendas con `Id` 524 y 1299, cuando cumplen las condiciones documentadas, **solo del conjunto de entrenamiento**.
- Evaluación del modelo y del `DummyRegressor` sobre la misma partición.
- Almacenamiento de métricas, metadatos y versiones del pipeline completo.
- Incorporación de pruebas automatizadas y un verificador integral de Docker.

## 2. Organización del repositorio

| Ruta | Contenido |
|---|---|
| `fase-1/` | Entrega anterior: notebook, datos, modelo y documentación |
| `fase-2/train.py` | Entrada de entrenamiento por línea de comandos |
| `fase-2/predict.py` | Entrada de predicción por línea de comandos |
| `fase-2/verify_docker.py` | Verificación integral automatizada de Docker |
| `fase-2/requirements.txt` | Dependencias con versiones explícitas |
| `fase-2/requirements-dev.txt` | Referencia a las dependencias del proyecto |
| `fase-2/Dockerfile` | Construcción de la imagen |
| `fase-2/.dockerignore` | Exclusiones del contexto de construcción |
| `fase-2/README.md` | Documentación de la Fase 2 |
| `fase-2/house_prices/` | Lógica reutilizable de la aplicación |
| `fase-2/data/example_prediction.csv` | Ejemplo de predicción sin variable objetivo |
| `fase-2/tests/test_pipeline.py` | Pruebas del pipeline y de la interfaz CLI |
| `fase-2/tests/test_requisitos_fase2.py` | Pruebas complementarias de los requisitos |
| `fase-2/models/entrega-v1/` | Modelo, métricas y metadatos de la primera versión |
| `fase-2/models/entrega-v2/` | Modelo, métricas y metadatos de la segunda versión |
| `fase-2/outputs/` | Resultados generados durante las ejecuciones |
| `pytest.ini` | Configuración de las pruebas |
| `.gitignore` | Exclusiones del control de versiones |

La carpeta `outputs/` se crea cuando se guardan predicciones.

Los scripts son interfaces de línea de comandos con `argparse`. Las responsabilidades del paquete `house_prices/` se distribuyen así:

| Archivo | Responsabilidad |
|---|---|
| `data.py` | Lectura y validación de datos y columnas |
| `preprocessing.py` | Transformaciones semánticas de las variables |
| `pipeline.py` | Construcción del pipeline de regresión |
| `model.py` | Partición, entrenamiento y evaluación |
| `storage.py` | Guardado y carga de versiones y artefactos |
| `prediction.py` | Validación, predicción y escritura del CSV de salida |

La carpeta `models/` puede contener versiones adicionales generadas para comprobaciones locales. Estas no sustituyen a `entrega-v1` y `entrega-v2`.

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
py -3.13 -m venv fase-2/.venv
```

Si Python 3.13 ya es el intérprete predeterminado, también puede utilizarse:

```powershell
python -m venv fase-2/.venv
```

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

### Esquema del CSV de predicción

Debe contener las columnas predictoras esperadas por la versión del modelo guardado, con tipos compatibles. **No debe incluir `SalePrice`.**

`GarageType` es una columna comprobada explícitamente durante la validación. `Id`, `SaleType` y `SaleCondition` son opcionales al predecir porque no son predictores del modelo; si aparece `Id`, se conserva en la salida.

- Los valores faltantes permitidos se tratan con el pipeline previamente ajustado; una celda nula no equivale a una columna obligatoria ausente.
- Las columnas se reordenan según el esquema guardado; las adicionales se ignoran.
- Las categorías nuevas se admiten mediante la codificación One-Hot configurada.
- Si falta una columna obligatoria, hay tipos incompatibles o no existe el modelo, se informa el error y no se genera una salida aparentemente válida.

### Salida

Se genera un CSV con `SalePricePrediction` y, cuando se incluye en la entrada, `Id`, manteniendo el orden y la cantidad de filas.

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
| `--seed` | Semilla para reproducir la partición; por defecto, 42 |
| `--test-size` | Proporción de evaluación; por defecto, 0.2 |

Con la configuración del ejemplo se realiza una partición aleatoria **80/20**. El preprocesamiento se ajusta exclusivamente con los datos de entrenamiento y se aplica al conjunto de prueba sin reajustarlo.

El comando también compara la regresión lineal con `DummyRegressor` y muestra las métricas y las rutas de los archivos generados.

La ejecución crea:

- `fase-2/models/nueva-v1/model.joblib`
- `fase-2/models/nueva-v1/metrics.json`
- `fase-2/models/nueva-v1/metadata.json`

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

**Limitaciones conocidas:** un R² cercano a 0,45 deja una parte importante de la variabilidad sin explicar; el resultado depende de los datos disponibles y del escenario de Ames. Entre equipos o bibliotecas de álgebra lineal pueden existir pequeñas diferencias numéricas, aunque se conserve la semilla 42.

En una comprobación previa documentada, dos entrenamientos con los mismos datos y configuración (`v1` y `comprobacion_punto8`) presentaron las mismas métricas y sus predicciones se compararon con `numpy.testing.assert_allclose` (`rtol=1e-10`, `atol=1e-8`).

## 8. Versionamiento y persistencia de modelos

Las dos versiones destinadas a la entrega son `entrega-v1` y `entrega-v2`, almacenadas en `fase-2/models/`.

Cada versión contiene:

| Archivo | Contenido |
|---|---|
| `model.joblib` | Pipeline y estimador ajustados |
| `metrics.json` | Métricas del modelo y del `DummyRegressor`, partición y parámetros de evaluación |
| `metadata.json` | Versión, fecha UTC, objetivo, algoritmo, parámetros, columnas, tipos, datos utilizados, cantidades de registros, semilla, partición, entorno y huellas SHA-256 |

Las versiones `entrega-v1` y `entrega-v2` se generaron con la misma semilla, datos y configuración, por lo que reportan las mismas métricas. Ambas pueden cargarse para predecir de manera independiente:

```powershell
python fase-2/predict.py --model fase-2/models/entrega-v1 --input fase-2/data/example_prediction.csv --output fase-2/outputs/entrega-v1.csv
python fase-2/predict.py --model fase-2/models/entrega-v2 --input fase-2/data/example_prediction.csv --output fase-2/outputs/entrega-v2.csv
```

Las versiones existentes no se sobrescriben silenciosamente. Los modelos, sus métricas y sus metadatos se conservan en carpetas independientes.

Los metadatos incluyen huellas SHA-256 de los datos, del código de la aplicación y de `requirements.txt`, además de la versión de Python y las bibliotecas utilizadas.

Las versiones temporales de práctica o validación no deben confundirse con las versiones de entrega.

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

### Ajuste de OpenBLAS en Windows

En la validación local del 3 de noviembre de 2026 se observó el mensaje `OpenBLAS error: Memory allocation still failed` al iniciar los subprocesos que comprueban la interfaz de línea de comandos.

La ejecución se completó limitando a un hilo las bibliotecas de cálculo numérico. Si aparece este error, ejecutar desde la **raíz del repositorio**, en la misma sesión de PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"

.\fase-2\.venv\Scripts\python.exe -m pytest fase-2/tests -q
```

Las variables afectan a esa sesión y a los procesos que se inicien desde ella. El resultado observado fue **13 pruebas aprobadas en 8,64 segundos**.

Este ajuste no modifica los datos, las pruebas ni los modelos guardados.

## 10. Ejecución con Docker

Docker Desktop debe estar iniciado y utilizando contenedores Linux. El `Dockerfile` usa `python:3.13-slim`, instala `requirements.txt` y copia los scripts y el paquete reutilizable.

El entrenamiento **no** se ejecuta durante `docker build`.

El `Dockerfile` configura `ENTRYPOINT ["python"]`; por tanto, en `docker run` se indica directamente `train.py` o `predict.py`, **sin escribir `python` nuevamente**.

**Los comandos de las secciones 10.1 a 10.6 se ejecutan desde la carpeta `fase-2` en PowerShell.** La sección 10.7 indica también la alternativa desde la raíz.

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

Las predicciones permanecen en `fase-2/outputs/predicciones_docker.csv`.

Para inspeccionar las primeras filas desde PowerShell, aún dentro de `fase-2`:

```powershell
Get-Content .\outputs\predicciones_docker.csv -TotalCount 6
```

Para usar el archivo de ejemplo en lugar de `test.csv`, se puede montar `data/example_prediction.csv` como entrada, adaptando la ruta de `--input`.

### 10.5. Ejecutar las pruebas con Docker

El Dockerfile copia los scripts y el paquete de la aplicación, sin incorporar la carpeta de pruebas. Para ejecutarlas dentro del contenedor, se montan temporalmente los archivos necesarios.

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

Se documentaron las siguientes operaciones dentro de Docker:

| Validación | Resultado |
|---|---|
| Construcción de la imagen | Correcta |
| Ayuda de `train.py` y `predict.py` | Disponible |
| Pruebas automatizadas | **13 aprobadas, 0 fallidas** |
| Entrenamiento desde `fase-1/train.csv` | Correcto |
| Persistencia de `model.joblib`, `metrics.json` y `metadata.json` | Correcta en carpeta del equipo |
| Predicción con `fase-1/test.csv` | Correcta |
| CSV `outputs/predicciones_docker.csv` | Generado y consultado correctamente |

La ejecución de entrenamiento en Docker produjo MAE = **21.182,45**, RMSE = **64.975,84** y R² = **0,4496** para regresión lineal.

Para `DummyRegressor`, los resultados fueron MAE = **62.580,40**, RMSE = **87.619,50** y R² = **-0,0009**.

La ejecución original de las pruebas en Docker mostró **13 aprobadas** y una advertencia sobre la caché en un directorio de solo lectura. La opción `-p no:cacheprovider` evita esa advertencia.

### 10.7. Verificación integral automatizada

`verify_docker.py` reúne la comprobación de Docker en un solo comando.

Requiere Docker Desktop iniciado y un intérprete de Python en el anfitrión. El script utiliza la biblioteca estándar y no necesita instalar las dependencias del modelo fuera del contenedor para esta comprobación.

Desde la **raíz del repositorio**, en PowerShell:

```powershell
python fase-2/verify_docker.py
```

Si la terminal ya está en **`fase-2`**:

```powershell
python verify_docker.py
```

El verificador realiza estas operaciones:

1. Construye la imagen `house-prices-fase2:entrega-2` con el Dockerfile de la fase.
2. Copia los datos de entrenamiento y el ejemplo de predicción a una carpeta temporal del anfitrión, montada en `/data`.
3. Entrena `docker-v1` con semilla 42 y `docker-v2` con semilla 43.
4. Comprueba que cada versión conserve `model.joblib`, `metrics.json` y `metadata.json` después de finalizar el contenedor con `--rm`.
5. Genera predicciones con ambas versiones y comprueba el CSV del ejemplo de una vivienda.
6. Intenta predecir con un modelo inexistente y comprueba el código de salida distinto de cero y la ausencia de un CSV de resultados.

La comprobación finalizó correctamente el **3 de noviembre de 2026**, en Windows con Docker Desktop y la imagen basada en Python 3.13.

El mensaje final fue:

```text
Docker: construcción, dos versiones, predicción y persistencia verificadas.
```

Los artefactos temporales se eliminan al terminar la comprobación. El script no modifica las versiones `entrega-v1` y `entrega-v2` del repositorio.

Para conservar archivos de una ejecución propia, utilizar los montajes de las secciones 10.3 y 10.4.

El verificador no sustituye las pruebas de pytest: ambas comprobaciones se ejecutan por separado.

## 11. Limitaciones y consideraciones de uso

- El modelo fue evaluado con datos de Ames, Iowa; su rendimiento puede cambiar con viviendas de otras zonas o periodos.
- Los resultados no reemplazan una valoración inmobiliaria profesional.
- `R² = 0,4496` muestra margen de mejora; no se busca en esta fase cambiar obligatoriamente el algoritmo ni mejorar la métrica.
- Para predecir se requiere el esquema de variables y los tipos esperados por el modelo guardado.
- `YrSold` y `MoSold` se interpretan como fecha prevista de venta, de acuerdo con el escenario de predicción.
- El archivo `model.joblib` debe cargarse únicamente desde fuentes confiables, dado que la deserialización de artefactos de terceros puede ejecutar código.
- La reproducibilidad requiere conservar los datos, la configuración y las dependencias. Pueden existir pequeñas diferencias numéricas entre sistemas.
- No se exige una API REST, monitoreo, MLflow, DVC, Docker Compose ni publicar la imagen Docker en la nube en esta fase.

## 12. Fuentes y referencias

- **Datos del proyecto:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).
- **Antecedentes y decisiones técnicas:** [`fase-1/README.md`](../fase-1/README.md) y `fase-1/notebook.ipynb`.
- **Especificaciones del trabajo:** *Especificación Técnica — Fase 2: Scripts y Docker*, Proyecto Integrador 2026-II, Modelos y Simulación de Sistemas I.
- **Bibliotecas utilizadas:** `scikit-learn`, `pandas`, `numpy`, `joblib` y `pytest`, en las versiones fijadas en `requirements.txt`.
- **Configuración de hilos de OpenBLAS:** [Runtime variables](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/).


## 13. Identificación de la entrega

La Fase 2 se integró en `main` mediante el Pull Request #21, el 3 de noviembre de 2026. El commit de integración es `025dde3`, identificado con la etiqueta `entrega-2`.

Las versiones `entrega-v1` y `entrega-v2` corresponden a los artefactos del modelo almacenados en `models/` y son independientes de la versión del código.

Los aportes y las revisiones de cada integrante se consultan en los commits y Pull Requests del repositorio.

