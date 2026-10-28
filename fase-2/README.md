# Fase 2 — Scripts y Docker

Esta fase conserva la regresión lineal y los datos de Ames usados en Fase 1,
pero permite entrenar y predecir desde la línea de comandos. Las transformaciones
semánticas y el preprocesamiento estadístico forman parte de un único pipeline
persistido con el modelo.

## Organización y comandos

Los puntos de entrada solo interpretan argumentos y coordinan módulos
reutilizables:

- `train.py` y `predict.py`: interfaces CLI con `argparse`; no leen datos ni
  entrenan al importarse. Consulta sus opciones con
  `python fase-2/train.py --help` y `python fase-2/predict.py --help`.
- `house_prices/data.py`: lectura de CSV y validación de columnas, filas y
  variable objetivo.
- `house_prices/pipeline.py` y `preprocessing.py`: construcción del pipeline y
  transformaciones compartidas por ajuste e inferencia.
- `house_prices/model.py`: partición, ajuste y evaluación.
- `house_prices/storage.py`: guardado versionado y carga de modelos.
- `house_prices/prediction.py`: predicción validada y escritura del CSV de
  salida.

## Preparación local

Desde la raíz del repositorio, instala las dependencias:

```bash
python -m pip install -r fase-2/requirements.txt
```

El archivo `fase-1/train.csv` debe estar disponible (se incluye en este
repositorio). Para evaluar el modelo con una partición reproducible 80/20 y
guardar una nueva versión, indica el nombre del objetivo, el directorio y una
versión que aún no exista:

```bash
python fase-2/train.py --data fase-1/train.csv --target SalePrice --output-dir fase-2/models --version v1 --seed 42 --test-size 0.2
```

`--target` debe nombrar una columna numérica del CSV (para Ames, `SalePrice`).
`--seed` y `--test-size` controlan la partición aleatoria reproducible. El
comando imprime las métricas de regresión lineal y `DummyRegressor` medidas
sobre el mismo holdout, además de las rutas generadas:

```text
fase-2/models/v1/model.joblib
fase-2/models/v1/metrics.json
fase-2/models/v1/metadata.json
```

La partición se realiza antes del preprocesamiento. Los outliers documentados
(Id 524 y 1299, con `SaleCondition == "Partial"` y `GrLivArea > 4000`) se
excluyen solo de entrenamiento. El pipeline guardado incorpora las
transformaciones ajustadas exclusivamente con ese conjunto de entrenamiento
filtrado; el holdout no participa en el ajuste. Cada versión debe ser única:
el comando falla si ya existe un directorio con ese identificador, para evitar
sobrescribir artefactos previos.

Para predecir con registros sin la variable objetivo. Puedes pasar el artefacto
`model.joblib` o directamente el directorio de la versión:

El repositorio incluye `fase-2/data/example_prediction.csv`, un ejemplo pequeño
con el esquema real del proyecto y sin `SalePrice`. También puedes proporcionar
otro CSV con las variables predictoras originales.

```bash
python fase-2/predict.py --model fase-2/models/v1 --input fase-2/data/example_prediction.csv --output fase-2/predictions.csv
```

La entrada debe incluir todas las columnas predictoras usadas en el
entrenamiento. `Id`, `SaleType` y `SaleCondition` son opcionales porque no se
usan como predictores; si están presentes, `Id` se conserva en la salida. Las
columnas adicionales se ignoran y las columnas se reordenan según el esquema
del pipeline guardado. Las categorías no observadas durante el entrenamiento
se aceptan y One-Hot las ignora. Los valores nulos se imputan con las
transformaciones guardadas; una columna obligatoria ausente o un valor de tipo
incompatible produce un error comprensible y no escribe resultados. La salida
contiene `SalePricePrediction` y, si estaba disponible, `Id`; si no había `Id`,
el orden de filas de la salida corresponde al de entrada.

## Ejecución con Docker

Construye la imagen desde la raíz del repositorio:

```bash
docker buildx build --load -t house-prices-fase2 -f fase-2/Dockerfile fase-2
```

En PowerShell, monta `fase-1` en `/data` para leer los datos y persistir modelos
y resultados en el equipo anfitrión:

```powershell
docker run --rm -v "${PWD}\fase-1:/data" house-prices-fase2 train.py `
  --data /data/train.csv --target SalePrice --output-dir /data/models `
  --version v1 --seed 42 --test-size 0.2
```

Usa la ruta de modelo que imprima el entrenamiento al generar predicciones:

```powershell
docker run --rm -v "${PWD}\fase-1:/data" house-prices-fase2 predict.py `
  --model /data/models/v1/model.joblib `
  --input /data/test.csv --output /data/predictions.csv
```

El volumen mantiene los modelos versionados y los CSV fuera del contenedor. No
se requiere Docker Compose.

## Preprocesamiento y momento de predicción

- `Id` se elimina del conjunto de predictores; se usa durante entrenamiento
  únicamente para reproducir el filtro documentado de los dos outliers.
- Las categorías de variables de espacios ausentes (por ejemplo, sótano,
  garaje o piscina) se representan con la categoría `"None"` y se añade
  `HasGarage` dentro del pipeline.
- Los ocho valores realmente faltantes de `MasVnrType` y `MasVnrArea` se
  conservan como desconocidos. No se convierten en ausencia de mampostería:
  la categoría desconocida se imputa con la categoría más frecuente y el área
  numérica con la mediana. Ambas estadísticas se aprenden únicamente al ajustar
  el pipeline con los datos de entrenamiento.
- Las variables numéricas se imputan con mediana y escalan; las categóricas se
  imputan con la categoría más frecuente y se codifican con One-Hot. Las
  categorías nuevas durante predicción se toleran.
- La predicción se plantea antes de concretar la venta. Se conservan `YrSold`
  y `MoSold` como fecha prevista, pero `SaleType` y `SaleCondition` se excluyen
  porque describen la transacción y no están disponibles al tasar/publicar la
  vivienda. `SalePrice` se usa exclusivamente como objetivo durante
  entrenamiento y se descarta si aparece en el CSV de predicción.
- La evaluación se realiza en un split aleatorio reproducible con semilla 42.
  Imputación, escalamiento y codificación se ajustan dentro del pipeline solo
  con el split de entrenamiento durante la evaluación.

## Pruebas

Instala las dependencias de desarrollo y ejecuta pytest:

```bash
python -m pip install -r fase-2/requirements-dev.txt
python -m pytest fase-2/tests
```

Las pruebas cubren las reglas semánticas, la distinción entre ausencia y dato
desconocido de mampostería, la validación del objetivo configurable, la
comparación con baseline, la persistencia de modelo/métricas/metadatos y
predicciones con registros nuevos.
