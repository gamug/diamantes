import numpy as np
import pandas as pd

from model.baseline import HeuristicPriceRegressor


def _synthetic_diamonds(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    carat = rng.uniform(0.3, 3.0, size=n)
    clarity = rng.choice(["I1", "SI1", "VVS1", "IF"], size=n)
    color = rng.choice(["J", "G", "D"], size=n)
    cut = rng.choice(["Fair", "Good", "Ideal"], size=n)

    clarity_mult = {"I1": 0.7, "SI1": 1.0, "VVS1": 1.3, "IF": 1.6}
    price = carat * 4000 * np.array([clarity_mult[c] for c in clarity])

    return pd.DataFrame(
        {"carat": carat, "clarity": clarity, "color": color, "cut": cut, "price": price}
    )


def test_fit_predict_learns_positive_prices() -> None:
    df = _synthetic_diamonds()
    x = df[["carat", "clarity", "color", "cut"]]
    y = df["price"]

    model = HeuristicPriceRegressor(n_carat_bins=5).fit(x, y)
    predictions = model.predict(x)

    assert predictions.shape == (len(df),)
    assert (predictions > 0).all()


def test_predict_ranks_higher_clarity_as_more_expensive() -> None:
    df = _synthetic_diamonds()
    x = df[["carat", "clarity", "color", "cut"]]
    y = df["price"]
    model = HeuristicPriceRegressor().fit(x, y)

    same_carat = pd.DataFrame(
        {
            "carat": [1.0, 1.0],
            "clarity": ["I1", "IF"],
            "color": ["G", "G"],
            "cut": ["Ideal", "Ideal"],
        }
    )
    predicted = model.predict(same_carat)
    assert predicted[1] > predicted[0]  # IF (best) should price above I1 (worst)


def test_predict_handles_unseen_categories_gracefully() -> None:
    df = _synthetic_diamonds()
    x = df[["carat", "clarity", "color", "cut"]]
    y = df["price"]
    model = HeuristicPriceRegressor().fit(x, y)

    unseen = pd.DataFrame(
        {"carat": [1.0], "clarity": ["VVS2"], "color": ["F"], "cut": ["Very Good"]}
    )
    predicted = model.predict(unseen)
    assert predicted.shape == (1,)
    assert np.isfinite(predicted).all()
