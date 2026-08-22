"""Model training: pipeline construction, train/test split and hyperparameter search.

Ported from ``notebooks/5-models/train_model_pipeline.py``, the script
version confirmed by ``notebooks/5-models/02-...model_selection`` and
``03-...first_model`` as the best-performing model family
(``HistGradientBoostingRegressor`` over a log-transformed target).
"""

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from data.features import build_preprocessor

#: Hyperparameter grid selected in
#: ``notebooks/5-models/02-gmg-basic_algorithms_model_selection-2026_08_18.ipynb``
#: and confirmed in ``03-gmg-first_model-2026_08_18.ipynb``.
DEFAULT_HYPERPARAMETER_GRID: dict[str, list[Any]] = {
    "regressor__model__max_iter": [200, 300, 400],
    "regressor__model__max_depth": [6, 10, None],
    "regressor__model__learning_rate": [0.03, 0.05, 0.1],
    "regressor__model__l2_regularization": [0.0, 1.0],
}

#: GridSearchCV scoring used to select the best hyperparameters, consistent
#: with MAPE being this project's primary evaluation metric.
DEFAULT_SCORING = "neg_mean_absolute_percentage_error"


def make_stratify_key(df: pd.DataFrame) -> pd.Series:
    """Build the ``clarity x cut`` key used to stratify the train/test split.

    ``clarity`` and ``cut`` are the two grades with the strongest effect on
    price-per-carat, and some grade combinations are rare (a handful of
    rows), so a plain random split risks leaving some combinations entirely
    out of train or test.

    Args:
        df: DataFrame containing ``clarity`` and ``cut`` columns.

    Returns:
        A string series combining both columns, e.g. ``"VS1_Ideal"``.
    """
    return df["clarity"].astype(str) + "_" + df["cut"].astype(str)


def split_train_test(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features/target into train and test sets, stratified by ``clarity x cut``.

    Args:
        x: Feature DataFrame; must contain ``clarity`` and ``cut`` columns.
        y: Target series, aligned with ``x``.
        test_size: Fraction of rows held out for testing.
        random_state: Seed for reproducibility.

    Returns:
        ``(x_train, x_test, y_train, y_test)``.
    """
    stratify_key = make_stratify_key(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=stratify_key
    )
    return (
        cast(pd.DataFrame, x_train),
        cast(pd.DataFrame, x_test),
        cast(pd.Series, y_train),
        cast(pd.Series, y_test),
    )


def build_model_pipeline(
    preprocessor: ColumnTransformer | None = None,
) -> TransformedTargetRegressor:
    """Build the full (untrained) diamond-price model pipeline.

    Preprocessing (see :func:`src.data.features.build_preprocessor`) feeds a
    ``HistGradientBoostingRegressor``. The whole regressor is wrapped in a
    ``TransformedTargetRegressor`` that fits on ``log(price)`` and inverts
    with ``exp`` at predict time — this linearizes the convex
    carat -> price relationship and guarantees positive predictions.

    Args:
        preprocessor: Defaults to :func:`src.data.features.build_preprocessor`'s
            output.

    Returns:
        An unfitted ``TransformedTargetRegressor``.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    return TransformedTargetRegressor(
        regressor=Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", HistGradientBoostingRegressor(random_state=42)),
            ]
        ),
        func=np.log,
        inverse_func=np.exp,
    )


def train_model(  # noqa: PLR0913 -- grid-search knobs are each independently meaningful to a caller
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    hyperparameters: dict[str, list[Any]] | None = None,
    cv: int = 5,
    scoring: str = DEFAULT_SCORING,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Tune and fit the diamond-price model pipeline via grid search.

    Args:
        x_train: Training features.
        y_train: Training target (price).
        hyperparameters: Grid to search. Defaults to
            :data:`DEFAULT_HYPERPARAMETER_GRID`.
        cv: Number of cross-validation folds.
        scoring: sklearn scoring string. Defaults to
            :data:`DEFAULT_SCORING`.
        n_jobs: Number of parallel jobs (``-1`` uses all cores).

    Returns:
        The fitted ``GridSearchCV``; use ``.best_estimator_`` for the trained
        pipeline, ``.best_params_``/``.best_score_`` for the search result.
    """
    if hyperparameters is None:
        hyperparameters = DEFAULT_HYPERPARAMETER_GRID

    pipeline = build_model_pipeline()
    grid_search = GridSearchCV(
        pipeline,
        hyperparameters,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    grid_search.fit(x_train, y_train)
    return grid_search
