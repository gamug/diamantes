from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from inference.predict import MODEL_FILE_NAME
from model.persistence import save_model
from pipelines.inference_pipeline.run import run


class _StubModel:
    def predict(self, x: pd.DataFrame) -> Any:
        return np.full(len(x), 500.0)


def test_inference_pipeline_run_writes_predictions(tmp_path: Path) -> None:
    model_dir = tmp_path / "06_models"
    save_model(_StubModel(), model_dir / MODEL_FILE_NAME)

    input_csv_path = tmp_path / "05_model_input" / "diamantes_new.csv"
    input_csv_path.parent.mkdir(parents=True)
    input_csv_path.write_text(
        "carat,cut,color,clarity,depth,table,x,y,z\n"
        "1.01,Ideal,G,VS2,61.8,57.0,6.42,6.45,3.98\n"
        "1.51,Premium,H,SI1,62.0,58.0,7.32,7.28,4.53\n"
    )
    output_csv_path = tmp_path / "07_model_output" / "diamantes_predictions.csv"

    result_df = run(
        input_csv_path=input_csv_path,
        output_csv_path=output_csv_path,
        model_dir=model_dir,
    )

    assert output_csv_path.exists()
    assert (result_df["Predicted_Price"] == 500.0).all()

    written_df = pd.read_csv(output_csv_path)
    assert "Predicted_Price" in written_df.columns
    assert len(written_df) == 2


def test_inference_pipeline_run_raises_on_missing_columns(tmp_path: Path) -> None:
    model_dir = tmp_path / "06_models"
    save_model(_StubModel(), model_dir / MODEL_FILE_NAME)

    input_csv_path = tmp_path / "diamantes_new.csv"
    input_csv_path.write_text("carat,cut\n1.01,Ideal\n")
    output_csv_path = tmp_path / "diamantes_predictions.csv"

    with pytest.raises(ValueError, match="missing required columns"):
        run(input_csv_path=input_csv_path, output_csv_path=output_csv_path, model_dir=model_dir)
