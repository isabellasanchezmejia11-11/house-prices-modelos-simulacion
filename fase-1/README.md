# Fase 1 — Modelo Predictivo: House Prices

Proyecto Semestral — Modelos y Simulación de Sistemas I

Universidad de Antioquia

2026-II

## Integrantes

- Nicole Mariana Adarve Tangarife.
- Isabella Sánchez Mejía.
- Maria Alejandra Otálvaro Ramírez.

## Descripción del problema

Este proyecto desarrolla un modelo predictivo para estimar el precio de
venta de viviendas residenciales en Ames, Iowa, a partir de sus
características físicas, estructurales, de ubicación y de calidad. Es un
problema de aprendizaje supervisado de tipo regresión, donde la variable
objetivo es `SalePrice` (precio de venta en dólares).

## Fuente del conjunto de datos

Competición de Kaggle "House Prices — Advanced Regression Techniques"
(https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
1460 observaciones y 81 variables (numéricas y categóricas) relacionadas
con la calidad, el estado, la ubicación y las dimensiones de las
viviendas.

## Objetivo del modelo

Predecir el precio de venta (`SalePrice`) de una vivienda a partir de sus
características, construyendo una solución reproducible y correctamente
documentada que sirva como base para las siguientes fases del proyecto
(scripts ejecutables, API REST y monitoreo).

## Algoritmo utilizado

Regresión Lineal (`LinearRegression` de scikit-learn), integrada dentro
de un `Pipeline` que combina:
- Imputación de valores faltantes (mediana para numéricas, categoría más
  frecuente para categóricas).
- Codificación One-Hot para variables categóricas.
- Escalamiento estándar para variables numéricas.

Se comparó contra un modelo base (`DummyRegressor`, estrategia `mean`)
como punto de referencia.

## Métrica empleada

Se utilizaron tres métricas de evaluación para problemas de regresión:
**MAE**, **RMSE** y **R²**, calculadas sobre el conjunto de prueba (20%
de los datos, separados con `random_state` fijo para reproducibilidad).

## Principales resultados

| Métrica | Modelo base (Dummy) | Regresión Lineal |
|---|---|---|
| MAE  | $62,575.93 | $21,108.07 |
| RMSE | $87,619.03 | $65,385.64 |
| R²   | -0.0009    | 0.4426     |

La Regresión Lineal mejora de forma considerable al modelo base en las
tres métricas: reduce el error absoluto promedio (MAE) en más de
$41,000, reduce el RMSE en aproximadamente $22,000, y pasa de explicar
prácticamente nada de la variabilidad del precio (R² ≈ 0) a explicar
cerca del **44.3%** de ella. Esto confirma que incorporar las
características de las viviendas aporta valor predictivo real frente a
simplemente estimar el precio promedio.

## Instrucciones para ejecutar el notebook

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/isabellasanchezmejia11-11/house-prices-modelos-simulacion.git
   ```
2. Descargar `train.csv` desde la competición de Kaggle
   (House Prices – Advanced Regression Techniques) y ubicarlo dentro de
   la carpeta `fase-1/`.
3. Instalar las dependencias con las versiones exactas utilizadas:
   ```bash
   pip install pandas==3.0.5 numpy==2.4.6 scikit-learn==1.9.0 joblib==1.6.0
   ```
4. Abrir `fase-1/notebook.ipynb` y ejecutar todas las celdas en orden
   (Kernel → Restart & Run All).
5. Al finalizar, el modelo entrenado quedará guardado en
   `fase-1/modelo.joblib`, listo para cargarse y generar nuevas
   predicciones sin necesidad de reentrenar.

## Dependencias (versiones exactas)

- pandas==3.0.5
- numpy==2.4.6
- scikit-learn==1.9.0
- joblib==1.6.0
