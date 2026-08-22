from typing import Any

import numpy as np
import pandas as pd

from inference.explain import baseline_price, explain_prediction
from model.train import split_train_test, train_model


def _fitted_model() -> Any:
    rng = np.random.RandomState(0)
    n = 40
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
    x = df.drop(columns="price")
    y = df["price"]
    x_train, _, y_train, _ = split_train_test(x, y)

    grid_search = train_model(
        x_train,
        y_train,
        hyperparameters={
            "regressor__model__max_iter": [20],
            "regressor__model__max_depth": [3],
        },
        cv=2,
        n_jobs=1,
    )
    return grid_search.best_estimator_


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
                "color": "D",
                "clarity": "IF",
            }
        ]
    )


def test_explain_prediction_returns_one_row_per_feature() -> None:
    model = _fitted_model()
    diamond = _sample_diamond()

    result = explain_prediction(model, diamond)

    preprocessor = model.regressor_.named_steps["preprocessor"]
    assert len(result) == len(preprocessor.get_feature_names_out())
    assert set(result.columns) == {"feature", "multiplier", "effect"}
    assert result["effect"].isin(["increases price", "decreases price"]).all()


def test_baseline_price_is_a_positive_float() -> None:
    model = _fitted_model()
    baseline = baseline_price(model, _sample_diamond())

    assert isinstance(baseline, float)
    assert baseline > 0
