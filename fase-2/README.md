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

El comando imprime MAE, RMSE y R² en el conjunto de prueba. El pipeline final
se vuelve a ajustar con todos los registros después de medir esas métricas y se
guarda como `house_prices_<fecha-UTC>.joblib` en el directorio indicado. Cada
ejecución crea una versión nueva; conserva la ruta impresa para identificar el
modelo que usarás.

Para predecir con registros sin la variable objetivo:

Prepara `fase-1/test.csv` descargando el archivo `test.csv` de la competición
de Kaggle, o crea un CSV con la misma estructura de variables predictoras de
`train.csv` (sin `SalePrice`). Este repositorio incluye `train.csv`, pero no
incluye el conjunto de prueba de Kaggle.

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
docker build -t house-prices-fase2 -f fase-2/Dockerfile fase-2
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

- `Id` se elimina del conjunto de predictores.
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
- La predicción se plantea en el momento de cierre/registro de la venta: `YrSold`,
  `MoSold`, `SaleType` y `SaleCondition` deben estar disponibles en ese momento.
  No se utilizan para estimar un precio previo a la venta. `SalePrice` se usa
  exclusivamente como objetivo durante entrenamiento y se descarta si aparece
  accidentalmente en el CSV de entrada para predecir.
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
