import joblib
import numpy as np
import pandas as pd

from house_prices.model import train_model
from house_prices.preprocessing import SemanticPreprocessor
from house_prices.prediction import predict_file


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": range(1, 11),
            "GarageType": ["Attchd", None, "Detchd", None, "Attchd"] * 2,
            "GarageYrBlt": [2003, np.nan, 1998, np.nan, 2001] * 2,
            "MasVnrType": ["BrkFace", "None", np.nan, "Stone", "None"] * 2,
            "MasVnrArea": [196.0, 0.0, np.nan, 120.0, 0.0] * 2,
            "OverallQual": [7, 5, 6, 8, 4] * 2,
            "SalePrice": [208500, 181000, 223500, 250000, 129500] * 2,
        }
    )


def test_semantic_preprocessing_keeps_masonry_unknown_distinct_from_absence():
    raw = sample_data().drop(columns=["SalePrice"])

    transformed = SemanticPreprocessor().fit_transform(raw)

    assert transformed.loc[1, "GarageType"] == "None"
    assert transformed.loc[1, "HasGarage"] == 0
    assert transformed.loc[0, "HasGarage"] == 1
    assert transformed.loc[2, "MasVnrType"] != "None"
    assert pd.isna(transformed.loc[2, "MasVnrType"])
    assert pd.isna(transformed.loc[2, "MasVnrArea"])
    assert transformed.loc[1, "MasVnrArea"] == 0
    assert "Id" not in transformed.columns


def test_training_saves_loadable_model_and_prediction_csv(tmp_path):
    data = sample_data()

    model_path, metrics = train_model(data, tmp_path / "models")

    assert model_path.is_file()
    assert model_path.stat().st_size > 0
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert all(np.isfinite(value) for value in metrics.values())

    loaded_model = joblib.load(model_path)
    inference_data = data.drop(columns=["SalePrice"]).iloc[:2]
    input_path = tmp_path / "inference.csv"
    output_path = tmp_path / "results" / "predictions.csv"
    inference_data.to_csv(input_path, index=False)

    predict_file(model_path, input_path, output_path)

    predictions = pd.read_csv(output_path)
    assert predictions.columns.tolist() == ["Id", "SalePricePrediction"]
    assert predictions["Id"].tolist() == [1, 2]
    assert predictions["SalePricePrediction"].notna().all()
    assert len(loaded_model.predict(inference_data)) == 2
