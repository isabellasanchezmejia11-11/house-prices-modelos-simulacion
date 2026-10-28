# Fase 2 — Scripts y Docker

Esta fase conserva la regresión lineal y los datos de Ames usados en Fase 1,
pero permite entrenar y predecir desde la línea de comandos. Las transformaciones
semánticas y el preprocesamiento estadístico forman parte de un único pipeline
persistido con el modelo.

## Preparación local

Desde la raíz del repositorio, instala las dependencias:

```bash
python -m pip install -r fase-2/requirements.txt
```

El archivo `fase-1/train.csv` debe estar disponible (se incluye en este
repositorio). Para evaluar el modelo con una partición reproducible 80/20 y
guardar una nueva versión:

```bash
python fase-2/train.py --data fase-1/train.csv --models-dir fase-2/models
```

El comando imprime MAE, RMSE y R² en el conjunto de prueba. La partición se
realiza antes del preprocesamiento; los outliers documentados (Id 524 y 1299,
con `SaleCondition == "Partial"` y `GrLivArea > 4000`) se excluyen solo de
entrenamiento. El pipeline guardado se ajusta exclusivamente con ese conjunto
de entrenamiento filtrado, sin usar el holdout. Se guarda como
`house_prices_<fecha-UTC>.joblib`; cada ejecución crea una versión nueva.
Conserva la ruta impresa para identificar el modelo que usarás.

Para predecir con registros sin la variable objetivo:

El repositorio incluye `fase-1/test.csv`, descargado de la competición de
Kaggle. 

```powershell
python fase-2/predict.py `
  --model fase-2/models/house_prices_<fecha-UTC>.joblib `
  --data fase-1/test.csv `
  --output fase-2/predictions.csv
```

En PowerShell, reemplaza `<fecha-UTC>` por el nombre generado por el
entrenamiento. El CSV de entrada debe tener las columnas originales del modelo,
excepto `SalePrice`. Si incluye `Id`, se conserva en la salida; `Id` no se usa
como predictor. La salida contiene `SalePricePrediction`.

## Ejecución con Docker

Construye la imagen desde la raíz del repositorio:

```bash
docker buildx build --load -t house-prices-fase2 -f fase-2/Dockerfile fase-2
```

En PowerShell, monta `fase-1` en `/data` para leer los datos y persistir modelos
y resultados en el equipo anfitrión:

```powershell
docker run --rm -v "${PWD}\fase-1:/data" house-prices-fase2 train.py `
  --data /data/train.csv --models-dir /data/models
```

Usa la ruta de modelo que imprima el entrenamiento al generar predicciones:

```powershell
docker run --rm -v "${PWD}\fase-1:/data" house-prices-fase2 predict.py `
  --model /data/models/house_prices_<fecha-UTC>.joblib `
  --data /data/test.csv --output /data/predictions.csv
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
desconocido de mampostería, la persistencia/carga de un modelo y predicciones
con registros nuevos.
