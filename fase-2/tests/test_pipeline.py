import subprocess
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from house_prices.data import validate_prediction_data, validate_training_data
from house_prices.model import train_model
from house_prices.preprocessing import SemanticPreprocessor
from house_prices.prediction import predict_file, write_predictions


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": range(1, 11),
            "GarageType": ["Attchd", None, "Detchd", None, "Attchd"] * 2,
            "GarageYrBlt": [2003, np.nan, 1998, np.nan, 2001] * 2,
            "MasVnrType": ["BrkFace", "None", np.nan, "Stone", "None"] * 2,
            "MasVnrArea": [196.0, 0.0, np.nan, 120.0, 0.0] * 2,
            "OverallQual": [7, 5, 6, 8, 4] * 2,
            "GrLivArea": [1710, 1262, 1786, 2198, 1362] * 2,
            "SaleType": ["WD", "New", "WD", "WD", "COD"] * 2,
            "SaleCondition": ["Normal", "Partial", "Normal", "Normal", "Abnorml"] * 2,
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
    assert "SaleType" not in transformed.columns
    assert "SaleCondition" not in transformed.columns


def test_training_saves_loadable_model_and_prediction_csv(tmp_path):
    data = sample_data()

    artifacts = train_model(
        data,
        tmp_path / "models",
        version="test-v1",
        target_column="SalePrice",
    )

    assert artifacts.model_path.is_file()
    assert artifacts.model_path.stat().st_size > 0
    assert artifacts.metrics_path.is_file()
    assert artifacts.metadata_path.is_file()
    metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert set(metrics["models"]) == {"linear_regression", "dummy_regressor"}
    for model_metrics in (
        metrics["models"]["linear_regression"],
        metrics["models"]["dummy_regressor"]["metrics"],
    ):
        assert set(model_metrics) == {"mae", "rmse", "r2"}
        assert all(np.isfinite(value) for value in model_metrics.values())
    assert metadata["version"] == "test-v1"
    assert metadata["target"] == "SalePrice"
    assert metadata["seed"] == 42
    assert metadata["artifacts"]["model"] == "model.joblib"

    loaded_model = joblib.load(artifacts.model_path)
    inference_data = data.drop(columns=["SalePrice"]).iloc[:2]
    input_path = tmp_path / "inference.csv"
    output_path = tmp_path / "results" / "predictions.csv"
    inference_data.to_csv(input_path, index=False)

    reordered_input = inference_data.drop(columns=["SaleType", "SaleCondition"])
    reordered_input["ExtraColumn"] = "ignored"
    reordered_input = reordered_input.iloc[:, ::-1]
    reordered_input.iloc[0, reordered_input.columns.get_loc("OverallQual")] = np.nan
    reordered_input.iloc[0, reordered_input.columns.get_loc("GarageType")] = "UnseenGarageType"
    reordered_input.to_csv(input_path, index=False)

    predict_file(artifacts.model_path.parent, input_path, output_path)

    predictions = pd.read_csv(output_path)
    assert predictions.columns.tolist() == ["Id", "SalePricePrediction"]
    assert predictions["Id"].tolist() == [1, 2]
    assert predictions["SalePricePrediction"].notna().all()
    assert len(loaded_model.predict(inference_data)) == 2


def test_prediction_rejects_missing_or_incompatible_columns_without_output(tmp_path):
    data = sample_data()
    artifacts = train_model(
        data,
        tmp_path / "models",
        version="prediction-validation",
        target_column="SalePrice",
    )
    output_path = tmp_path / "predictions.csv"

    missing_input = data.drop(columns=["SalePrice", "OverallQual"])
    with pytest.raises(ValueError, match="OverallQual"):
        write_predictions(
            joblib.load(artifacts.model_path),
            missing_input.iloc[:1],
            output_path,
        )
    assert not output_path.exists()

    incompatible_input = data.drop(columns=["SalePrice"]).iloc[:1].copy()
    incompatible_input["OverallQual"] = "not-a-number"
    with pytest.raises(ValueError, match="OverallQual.*no numéricos"):
        write_predictions(
            joblib.load(artifacts.model_path),
            incompatible_input,
            output_path,
        )
    assert not output_path.exists()


def test_predict_cli_reports_missing_model_without_creating_output(tmp_path):
    repository_root = Path(__file__).resolve().parents[2]
    script_path = repository_root / "fase-2" / "predict.py"
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "predictions.csv"
    sample_data().drop(columns=["SalePrice"]).iloc[:1].to_csv(input_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--model",
            str(tmp_path / "missing.joblib"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "No se encontró el archivo del modelo" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output_path.exists()


def test_training_supports_configurable_target_and_refuses_version_overwrite(tmp_path):
    data = sample_data().rename(columns={"SalePrice": "TargetPrice"})
    output_dir = tmp_path / "models"
    artifacts = train_model(
        data,
        output_dir,
        version="v1",
        target_column="TargetPrice",
    )

    assert artifacts.model_path == output_dir / "v1" / "model.joblib"
    assert json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))["target"] == "TargetPrice"
    with pytest.raises(FileExistsError):
        train_model(data, output_dir, version="v1", target_column="TargetPrice")


def test_training_validation_reports_missing_required_columns():
    incomplete = sample_data().drop(columns=["SaleCondition"])

    with pytest.raises(ValueError, match="SaleCondition"):
        validate_training_data(incomplete)


def test_prediction_validation_rejects_target_column():
    inference_data = sample_data().drop(columns=["SalePrice"]).assign(SalePrice=123)

    with pytest.raises(ValueError, match="no debe incluir 'SalePrice'"):
        validate_prediction_data(inference_data)


@pytest.mark.parametrize("script", ["train.py", "predict.py"])
def test_cli_help_is_available_without_running_a_workflow(script):
    repository_root = Path(__file__).resolve().parents[2]
    script_path = repository_root / "fase-2" / script

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    if script == "predict.py":
        assert "--input" in result.stdout


def test_training_cli_help_documents_required_versioned_inputs():
    repository_root = Path(__file__).resolve().parents[2]
    script_path = repository_root / "fase-2" / "train.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    for argument in ("--data", "--target", "--output-dir", "--version", "--seed"):
        assert argument in result.stdout
