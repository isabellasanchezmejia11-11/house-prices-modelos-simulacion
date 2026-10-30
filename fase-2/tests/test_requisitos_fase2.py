import numpy as np
import pandas as pd
import pytest
import joblib


from house_prices.model import train_model
from house_prices.prediction import predict_file
from house_prices.pipeline import build_model

def test_prediccion_con_valores_faltantes(tmp_path):
    # 1. Crear datos sintéticos con el esquema requerido
    datos = pd.DataFrame({
        "Id": list(range(1, 11)),
        "OverallQual": [5, 6, 7, 8, 4, 6, 7, 5, 8, 6],
        "GrLivArea": [1000, 1200, 1500, 1800, 900,
                      1300, 1600, 1100, 1900, 1400],
        "GarageType": [
            "Attchd", "Detchd", "Attchd", "Attchd", "Detchd",
            "Attchd", "Detchd", "Attchd", "Attchd", "Detchd"
        ],
        "GarageYrBlt": [
            2000, 2005, np.nan, 2010, 1998,
            2003, 2008, np.nan, 2015, 2006
        ],
        "SaleCondition": ["Normal"] * 10,
        "SalePrice": [
            120000, 150000, 190000, 240000, 100000,
            165000, 205000, 135000, 260000, 180000
        ],
    })

    # 2. Entrenar y guardar un modelo temporal
    artefactos = train_model(
        datos,
        tmp_path / "models",
        version="test-nulos",
        target_column="SalePrice",
    )

    # 3. Preparar dos viviendas para predicción
    viviendas = datos.drop(columns=["SalePrice"]).iloc[:2].copy()
    viviendas.loc[0, "GarageYrBlt"] = np.nan

    entrada = tmp_path / "entrada.csv"
    salida = tmp_path / "predicciones.csv"

    viviendas.to_csv(entrada, index=False)

    # 4. Cargar el modelo guardado y predecir
    predict_file(artefactos.model_path, entrada, salida)

    # 5. Verificar los resultados
    assert salida.exists()

    resultado = pd.read_csv(salida)

    assert len(resultado) == len(viviendas)
    assert resultado["Id"].tolist() == viviendas["Id"].tolist()
    assert resultado["SalePricePrediction"].notna().all()
    assert np.isfinite(resultado["SalePricePrediction"]).all()



def test_versionamiento_de_modelos(tmp_path):
    # Crear datos sintéticos para entrenamiento
    datos = pd.DataFrame({
        "Id": list(range(1, 11)),
        "OverallQual": [5, 6, 7, 8, 4, 6, 7, 5, 8, 6],
        "GrLivArea": [1000, 1200, 1500, 1800, 900,
                      1300, 1600, 1100, 1900, 1400],
        "GarageType": [
            "Attchd", "Detchd", "Attchd", "Attchd", "Detchd",
            "Attchd", "Detchd", "Attchd", "Attchd", "Detchd"
        ],
        "GarageYrBlt": [
            2000, 2005, np.nan, 2010, 1998,
            2003, 2008, np.nan, 2015, 2006
        ],
        "SaleCondition": ["Normal"] * 10,
        "SalePrice": [
            120000, 150000, 190000, 240000, 100000,
            165000, 205000, 135000, 260000, 180000
        ],
    })

    carpeta_modelos = tmp_path / "models"

    # Crear primera versión
    version_1 = train_model(
        datos,
        carpeta_modelos,
        version="v1",
        target_column="SalePrice",
    )

    # Crear segunda versión
    version_2 = train_model(
        datos,
        carpeta_modelos,
        version="v2",
        target_column="SalePrice",
    )

    # Verificar que ambas versiones existen
    assert version_1.model_path.exists()
    assert version_2.model_path.exists()

    # Cargar ambas versiones
    modelo_1 = joblib.load(version_1.model_path)
    modelo_2 = joblib.load(version_2.model_path)

    assert modelo_1 is not None
    assert modelo_2 is not None

    contenido_original = version_1.model_path.read_bytes()

    # Verificar que una versión repetida sea rechazada
    with pytest.raises(FileExistsError):
        train_model(
            datos,
            carpeta_modelos,
            version="v1",
            target_column="SalePrice",
        )

    assert version_1.model_path.read_bytes() == contenido_original

    # Comprobar que ambas versiones siguen existiendo
    assert version_1.model_path.exists()
    assert version_2.model_path.exists()



def test_predicciones_se_conservan_al_guardar_modelo(tmp_path):
    # 1. Crear datos sintéticos
    datos = pd.DataFrame({
        "Id": list(range(1, 11)),
        "OverallQual": [5, 6, 7, 8, 4, 6, 7, 5, 8, 6],
        "GrLivArea": [1000, 1200, 1500, 1800, 900,
                      1300, 1600, 1100, 1900, 1400],
        "GarageType": [
            "Attchd", "Detchd", "Attchd", "Attchd", "Detchd",
            "Attchd", "Detchd", "Attchd", "Attchd", "Detchd"
        ],
        "GarageYrBlt": [
            2000, 2005, np.nan, 2010, 1998,
            2003, 2008, np.nan, 2015, 2006
        ],
        "SaleCondition": ["Normal"] * 10,
        "SalePrice": [
            120000, 150000, 190000, 240000, 100000,
            165000, 205000, 135000, 260000, 180000
        ],
    })

    # 2. Separar características y precio
    X = datos.drop(columns=["SalePrice"])
    y = datos["SalePrice"]

    # 3. Construir y entrenar el pipeline real
    modelo = build_model()
    modelo.fit(X, y)

    # 4. Predecir ANTES de guardar
    viviendas = X.iloc[:3].copy()
    predicciones_antes = modelo.predict(viviendas)

    # 5. Guardar el modelo temporalmente
    ruta_modelo = tmp_path / "modelo.joblib"
    joblib.dump(modelo, ruta_modelo)

    assert ruta_modelo.exists()

    # 6. Cargar el modelo guardado
    modelo_cargado = joblib.load(ruta_modelo)

    # 7. Predecir DESPUÉS de cargar
    predicciones_despues = modelo_cargado.predict(viviendas)

    # 8. Comparar ambas predicciones
    np.testing.assert_allclose(
        predicciones_antes,
        predicciones_despues,
        rtol=1e-6,
        atol=1e-6,
    )

    # 9. Verificar cantidad y valores válidos
    assert len(predicciones_despues) == len(viviendas)
    assert np.isfinite(predicciones_despues).all()
