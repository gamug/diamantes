from pathlib import Path

import numpy as np
import pandas as pd

from data.extraction import save_parquet
from model.persistence import load_model
from pipelines.training_pipeline.run import run


def _synthetic_feature_parquet(path: Path, n: int = 40, seed: int = 42) -> None:
    rng = np.random.RandomState(seed)
    carat = rng.uniform(0.3, 3.0, size=n)
    df = pd.DataFrame(
        {
            "carat": carat,
            "depth": rng.uniform(58.0, 65.0, size=n),
            "table": rng.uniform(54.0, 60.0, size=n),
            "x": carat * 2 + 4,
            "y": carat * 2 + 4,
            "z": carat + 2,
            "cut": np.where(rng.random(n) < 0.5, "Good", "Ideal"),
            "color": np.where(rng.random(n) < 0.5, "D", "J"),
            "clarity": np.where(rng.random(n) < 0.5, "SI1", "IF"),
            "price": carat * 4000 + rng.normal(0, 10, size=n),
        }
    )
    save_parquet(df, path)


def test_training_pipeline_run_saves_model_and_returns_metrics(tmp_path: Path) -> None:
    feature_path = tmp_path / "diamantes_clean.parquet"
    _synthetic_feature_parquet(feature_path)
    model_dir = tmp_path / "06_models"

    metrics = run(
        feature_parquet_path=feature_path,
        model_dir=model_dir,
        model_file_name="test-model.joblib",
        hyperparameters={
            "regressor__model__max_iter": [20],
            "regressor__model__max_depth": [3],
        },
        cv=2,
    )

    assert set(metrics.keys()) == {"MAE", "RMSE", "MAPE_%", "R2"}
    assert (model_dir / "test-model.joblib").exists()

    loaded_model = load_model(model_dir / "test-model.joblib")
    assert hasattr(loaded_model, "predict")
