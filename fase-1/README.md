# Fase 1 — Modelo predictivo: House Prices

Proyecto Integrador — Modelos y Simulación de Sistemas I
Universidad de Antioquia — 2026-II

## Integrantes

- Nicole Mariana Adarve Tangarife.
- Isabella Sánchez Mejía.
- Maria Alejandra Otálvaro Ramírez.

## Problema y datos

El proyecto estima el precio de venta `SalePrice` de viviendas residenciales
de Ames, Iowa, a partir de sus características físicas, estructurales, de
ubicación y calidad. Es un problema supervisado de regresión basado en la
competición [House Prices — Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
El conjunto de entrenamiento contiene 1.460 viviendas.

## Modelo y preprocesamiento

Se utiliza Regresión Lineal (`LinearRegression`) dentro de un pipeline y se
compara con `DummyRegressor` (estrategia `mean`). El pipeline aplica las reglas
semánticas de ausencia, deriva `HasGarage`, prepara `GarageYrBlt`, imputa
numéricas con la mediana y categóricas con la moda, escala las numéricas y
codifica las categorías con One-Hot. Las reglas semánticas y los imputadores
se aplican dentro del flujo reutilizable, tanto al entrenar como al predecir.

Los ocho faltantes reales de `MasVnrType` y `MasVnrArea` se conservan como
desconocidos: no se asume que representen ausencia de mampostería o área cero.
La categoría desconocida se imputa con la moda y el área con la mediana,
aprendidas únicamente durante el ajuste con los datos de entrenamiento.

`SaleType` y `SaleCondition` se excluyen de las variables predictoras porque
el escenario de uso es estimar el precio antes de que ocurra la venta. Se
conservan `MoSold` y `YrSold` como fecha prevista de venta. `SaleCondition`
solo se utiliza para identificar los registros atípicos documentados; no se
usa como predictor.

## Partición, evaluación y resultados

Se usa una partición aleatoria 80/20 con `random_state=42`. Las viviendas con
`Id` 524 y 1299 (ventas `Partial` con `GrLivArea > 4000`) se excluyen solo del
conjunto de entrenamiento. El split se realiza antes de ajustar el pipeline;
las estadísticas de imputación, escalamiento y codificación se aprenden solo
con train. El modelo persistido corresponde al pipeline ajustado con ese
conjunto de entrenamiento filtrado.

| Métrica | DummyRegressor | Regresión lineal |
|---|---:|---:|
| MAE | $62,580.40 | $21,182.45 |
| RMSE | $87,619.50 | $64,975.84 |
| R² | -0.0009 | 0.4496 |

Como análisis del momento de predicción, el notebook también compara un
pipeline que usa `SaleType` y `SaleCondition` con el modelo principal que las
excluye. Sin esas variables la evaluación obtiene MAE = $21,182.45,
RMSE = $64,975.84 y R² = 0.4496. Además de la mejora observada, excluirlas
evita depender de información que no está disponible antes de concretar la
venta.

## Ejecución del notebook

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/isabellasanchezmejia11-11/house-prices-modelos-simulacion.git
   ```
2. Asegurarse de que `train.csv` esté en `fase-1/`. El repositorio incluye el
   conjunto `test.csv` de Kaggle para generar predicciones.
3. Instalar las dependencias utilizadas:
   ```bash
   python -m pip install pandas==3.0.5 numpy==2.4.6 scikit-learn==1.9.0 joblib==1.6.0 matplotlib==3.11.1 seaborn==0.13.2
   ```
4. Desde `fase-1/`, abrir `notebook.ipynb` y ejecutar **Restart & Run All**.

El notebook guarda el pipeline entrenado en `fase-1/modelo.joblib`. El modelo
acepta las variables predictoras originales; descarta `Id`, `SalePrice` si
está presente, `SaleType` y `SaleCondition`, y calcula `HasGarage` durante el
preprocesamiento.

## Dependencias probadas

- pandas==3.0.5
- numpy==2.4.6
- scikit-learn==1.9.0
- joblib==1.6.0
- matplotlib==3.11.1
- seaborn==0.13.2
