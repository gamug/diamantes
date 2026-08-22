from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.constants import FEATURE_COLUMNS
from inference.predict import (
    MODEL_FILE_NAME,
    is_out_of_training_range,
    load_production_model,
    predict_price,
    preprocess_batch_data,
)
from model.persistence import save_model


class _StubModel:
    """Minimal stand-in for a fitted sklearn estimator."""

    def predict(self, x: pd.DataFrame) -> Any:
        assert list(x.columns) == FEATURE_COLUMNS
        return np.full(len(x), 999.0)


def _sample_diamond() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "carat": 1.0,
                "depth": 61.5,
                "table": 57.0,
                "x": 6.4,
                "y": 6.5,
                "z": 4.0,
                "cut": "Ideal",
                "color": "G",
                "clarity": "SI1",
            }
        ]
    )


def test_predict_price_selects_feature_columns_in_order() -> None:
    diamond = _sample_diamond()[list(reversed(FEATURE_COLUMNS))]  # shuffled column order
    predictions = predict_price(_StubModel(), diamond)
    assert predictions.tolist() == [999.0]


def test_is_out_of_training_range() -> None:
    diamonds = pd.DataFrame({"carat": [0.5, 1.0, 2.5]})
    result = is_out_of_training_range(diamonds)
    assert result.tolist() == [True, False, False]


def test_preprocess_batch_data_normalizes_columns_and_grades() -> None:
    df = pd.DataFrame(
        {
            " Carat ": ["1.01", "0.5x"],
            "Cut": [" ideal ", "premium"],
            "Color": ["g", "H"],
            "Clarity": ["si1", "VS2"],
        }
    )
    processed = preprocess_batch_data(df)

    assert list(processed.columns) == ["carat", "cut", "color", "clarity"]
    assert processed["cut"].tolist() == ["Ideal", "Premium"]
    assert processed["color"].tolist() == ["G", "H"]
    assert processed["clarity"].tolist() == ["SI1", "VS2"]
    assert processed["carat"].iloc[0] == 1.01
    assert pd.isna(processed["carat"].iloc[1])  # "0.5x" -> NaN, coerced not raised


def test_load_production_model_roundtrip(tmp_path: Path) -> None:
    save_model(_StubModel(), tmp_path / MODEL_FILE_NAME)
    loaded = load_production_model(tmp_path)
    predictions = predict_price(loaded, _sample_diamond())
    assert predictions.tolist() == [999.0]
