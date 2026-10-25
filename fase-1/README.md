# Fase 1 — Predicción de precios de viviendas

Proyecto de regresión lineal para estimar `SalePrice` en el conjunto House
Prices de Ames. El pipeline incluye preparación de datos, imputación,
escalamiento y codificación; se compara con `DummyRegressor` como baseline.

## Resultados

Evaluación sobre el 20 % de prueba (`random_state=42`). Los outliers de
`GrLivArea` con `Id` 524 y 1299 se excluyen solo del entrenamiento.

| Métrica | DummyRegressor | Regresión lineal |
|---|---:|---:|
| MAE | $62,580.40 | $21,182.45 |
| RMSE | $87,619.50 | $64,975.84 |
| R² | -0.0009 | 0.4496 |

## Ejecución

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/isabellasanchezmejia11-11/house-prices-modelos-simulacion.git
   ```
2. Descargar `train.csv` de la competición House Prices de Kaggle y guardarlo
   en `fase-1/`.
3. Instalar las dependencias:
   ```bash
   pip install pandas==3.0.5 numpy==2.4.6 scikit-learn==1.9.0 joblib==1.6.0 matplotlib==3.11.1 seaborn==0.13.2
   ```
4. Abrir `fase-1/notebook.ipynb` y ejecutar **Restart & Run All**.

El notebook guarda el pipeline reutilizable en `fase-1/modelo.joblib`. Recibe
un DataFrame con los predictores originales, sin `Id` ni `SalePrice`.

## Dependencias probadas

- pandas==3.0.5
- numpy==2.4.6
- scikit-learn==1.9.0
- joblib==1.6.0
- matplotlib==3.11.1
- seaborn==0.13.2
