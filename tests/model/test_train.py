import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor

from model.train import build_model_pipeline, make_stratify_key, split_train_test, train_model


def _synthetic_dataset(n: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    carat = rng.uniform(0.3, 3.0, size=n)
    depth = rng.uniform(58.0, 65.0, size=n)
    table = rng.uniform(54.0, 60.0, size=n)
    x = carat * 2 + rng.normal(0, 0.1, size=n) + 4
    y = carat * 2 + rng.normal(0, 0.1, size=n) + 4
    z = carat + rng.normal(0, 0.1, size=n) + 2
    # only 2 distinct clarity x cut combinations, so a 0.2 test split has enough members
    clarity = np.where(rng.random(n) < 0.5, "SI1", "IF")
    cut = np.where(rng.random(n) < 0.5, "Good", "Ideal")
    color = rng.choice(["D", "G", "J"], size=n)
    price = carat * 4000 + rng.normal(0, 50, size=n)

    return pd.DataFrame(
        {
            "carat": carat,
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z,
            "cut": cut,
            "color": color,
            "clarity": clarity,
            "price": price,
        }
    )


def test_make_stratify_key() -> None:
    df = pd.DataFrame({"clarity": ["VS1", "IF"], "cut": ["Ideal", "Good"]})
    key = make_stratify_key(df)
    assert list(key) == ["VS1_Ideal", "IF_Good"]


def test_split_train_test_preserves_row_count() -> None:
    df = _synthetic_dataset()
    x = df.drop(columns="price")
    y = df["price"]

    x_train, x_test, y_train, y_test = split_train_test(x, y, test_size=0.25, random_state=0)

    assert len(x_train) + len(x_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert len(x_train) == len(y_train)


def test_build_model_pipeline_is_transformed_target_regressor() -> None:
    pipeline = build_model_pipeline()
    assert isinstance(pipeline, TransformedTargetRegressor)
    assert pipeline.func is np.log
    assert pipeline.inverse_func is np.exp


def test_train_model_returns_fitted_grid_search() -> None:
    df = _synthetic_dataset()
    x = df.drop(columns="price")
    y = df["price"]
    x_train, _, y_train, _ = split_train_test(x, y)

    tiny_grid = {
        "regressor__model__max_iter": [20],
        "regressor__model__max_depth": [3],
    }
    grid_search = train_model(x_train, y_train, hyperparameters=tiny_grid, cv=2, n_jobs=1)

    assert hasattr(grid_search, "best_estimator_")
    predictions = grid_search.best_estimator_.predict(x_train)
    assert len(predictions) == len(x_train)
