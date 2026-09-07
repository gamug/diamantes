"""Training pipeline: feature table -> tuned diamond-price model + metrics.

Standalone, autonomously runnable script for issue #24 ("3. Training pipeline
development"), part of the FTI scripts architecture (issue #15).

It:

1. **reads** the processed feature table written by the feature pipeline
   (``data/04_feature/diamantes_features.parquet`` -- issue #22);
2. **splits** it into train / test sets (stratified on ``clarity`` x ``cut`` so
   rare grade combinations stay represented on both sides);
3. **trains** a :class:`~sklearn.ensemble.HistGradientBoostingRegressor` on a
   log-price target, tuning it with :class:`~sklearn.model_selection.GridSearchCV`
   (model / grid chosen in
   ``notebooks/5-models/02-gmg-basic_algorithms_model_selection-2026_08_18.ipynb``);
4. **evaluates** the tuned pipeline on the held-out test set (MAPE, RMSE, MAE,
   R-squared);
5. **stores** the fitted pipeline
   (``data/06_models/diamantes_price-hist_gradient_boosting-v1.joblib``) and the
   evaluation metrics (``data/08_reporting/training_metrics.json``).

Run it (``src`` is on ``sys.path`` via ``pyproject.toml``'s
``[tool.pytest.ini_options] pythonpath``; for a bare ``python`` call, either
``cd src`` or set ``PYTHONPATH=src``)::

    uv run python src/training_pipeline.py
    uv run python -m training_pipeline
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from diamond_features import KNOWN_CATEGORIES, PRICE_COLUMN, VOLUME_COLUMN
from feature_pipeline import FEATURE_PARQUET_PATH
from split_validation import validate_train_test_split

logger = logging.getLogger(__name__)

#: Repository root: this file lives at ``src/training_pipeline.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
#: Fitted model artifact (feature layer 06_models).
MODEL_PATH = DATA_DIR / "06_models" / "diamantes_price-hist_gradient_boosting-v1.joblib"
#: Evaluation results (reporting layer 08_reporting).
METRICS_PATH = DATA_DIR / "08_reporting" / "training_metrics.json"

#: Numeric predictors kept for the model (raw ``x``/``y``/``z`` are dropped --
#: near-collinear with ``carat`` and already folded into ``volume``, r up to 0.98
#: per the EDA).
NUMERIC_FEATURES: list[str] = ["carat", "depth", "table", VOLUME_COLUMN]
#: Ordinal categorical predictors, encoded worst -> best via
#: :data:`diamond_features.KNOWN_CATEGORIES`.
CATEGORICAL_FEATURES: list[str] = ["cut", "color", "clarity"]
#: Predictor columns, fixed order (numeric block then categorical block).
MODEL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
#: Regression target.
TARGET: str = PRICE_COLUMN

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5
N_JOBS: int = -1
#: Smallest ``clarity`` x ``cut`` group that still allows a stratified split.
MIN_STRATUM_SIZE: int = 2
#: GridSearchCV scoring key (percentage error, the business-facing metric).
SCORING: str = "neg_mean_absolute_percentage_error"

#: Default hyper-parameter grid. A reduced version of the notebook grid so the
#: script runs autonomously in a few minutes; pass ``param_grid={}`` to skip
#: tuning entirely and fit the base pipeline.
DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "regressor__model__max_iter": [200, 400],
    "regressor__model__max_depth": [6, None],
    "regressor__model__learning_rate": [0.05, 0.1],
}


def load_feature_table(path: Path = FEATURE_PARQUET_PATH) -> pd.DataFrame:
    """Load the feature table written by the feature pipeline.

    Args:
        path: Path to ``data/04_feature/diamantes_features.parquet``.

    Returns:
        The feature table, restricted to :data:`MODEL_FEATURES` + :data:`TARGET`.

    Raises:
        FileNotFoundError: If ``path`` does not exist -- run the feature
            pipeline first.
        ValueError: If the file is missing a required model column.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature table not found: {path}. Run the feature pipeline (issue #22) first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    missing = [column for column in [*MODEL_FEATURES, TARGET] if column not in df.columns]
    if missing:
        raise ValueError(f"feature table {path} is missing required columns: {missing}")
    return df[[*MODEL_FEATURES, TARGET]]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature table into the predictor matrix and the target vector.

    Args:
        df: Feature table from :func:`load_feature_table`.

    Returns:
        ``(X, y)`` where ``X`` has the :data:`MODEL_FEATURES` columns and ``y``
        is :data:`TARGET`.
    """
    return df[MODEL_FEATURES], df[TARGET]


def split_train_test(
    features: pd.DataFrame, target: pd.Series, *, stratify: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold out a test set, stratified on ``clarity`` x ``cut`` grade pairs.

    Some grade combinations are rare, so a plain random split risks under- or
    over-representing them between train and test.

    Args:
        features: Predictor matrix.
        target: Target vector.
        stratify: When ``True`` (default) stratify on the ``clarity`` x ``cut``
            key; disable for tiny datasets where some classes have one member.

    Returns:
        ``(x_train, x_test, y_train, y_test)``.
    """
    strata = None
    if stratify:
        strata = features["clarity"].astype(str).str.cat(features["cut"].astype(str), sep="_")
        if strata.value_counts().min() < MIN_STRATUM_SIZE:
            logger.warning(
                "Some clarity x cut grade pairs have a single sample; "
                "falling back to a non-stratified split."
            )
            strata = None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=strata,
    )
    return x_train, x_test, y_train, y_test


def build_model_pipeline(random_state: int = RANDOM_STATE) -> TransformedTargetRegressor:
    """Assemble the preprocessing + model pipeline.

    Numeric columns are median-imputed and standardised; categoricals are
    most-frequent-imputed and ordinal-encoded worst -> best. The regressor sees a
    ``log(price)`` target (:func:`numpy.log` / :func:`numpy.exp`), which
    linearises the convex carat -> price curve and keeps predictions positive.

    Args:
        random_state: Seed forwarded to the gradient-boosting regressor.

    Returns:
        An unfitted :class:`~sklearn.compose.TransformedTargetRegressor`.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[KNOWN_CATEGORIES[column] for column in CATEGORICAL_FEATURES],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    return TransformedTargetRegressor(
        regressor=Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", HistGradientBoostingRegressor(random_state=random_state)),
            ]
        ),
        func=np.log,
        inverse_func=np.exp,
    )


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    param_grid: dict[str, list[Any]] | None = None,
) -> TransformedTargetRegressor:
    """Fit the model pipeline, tuning hyper-parameters when a grid is given.

    Args:
        x_train: Training predictors.
        y_train: Training target.
        param_grid: Hyper-parameter grid for
            :class:`~sklearn.model_selection.GridSearchCV`. ``None`` uses
            :data:`DEFAULT_PARAM_GRID`; an empty dict skips tuning and fits the
            base pipeline directly.

    Returns:
        The fitted estimator (the best one found when tuning).
    """
    grid = DEFAULT_PARAM_GRID if param_grid is None else param_grid
    pipeline = build_model_pipeline()
    if not grid:
        pipeline.fit(x_train, y_train)
        return pipeline
    search = GridSearchCV(pipeline, grid, cv=CV_FOLDS, scoring=SCORING, n_jobs=N_JOBS)
    search.fit(x_train, y_train)
    logger.info("Best hyper-parameters: %s", search.best_params_)
    best_estimator: TransformedTargetRegressor = search.best_estimator_
    return best_estimator


def evaluate_model(
    model: TransformedTargetRegressor, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Score a fitted model on the held-out test set.

    Args:
        model: Fitted estimator from :func:`train_model`.
        x_test: Test predictors.
        y_test: Test target.

    Returns:
        ``{"mape_pct", "rmse", "mae", "r2", "n_test_samples"}``.
    """
    predictions = model.predict(x_test)
    return {
        "mape_pct": float(mean_absolute_percentage_error(y_test, predictions) * 100.0),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "n_test_samples": float(len(y_test)),
    }


def save_model(model: TransformedTargetRegressor, path: Path = MODEL_PATH) -> None:
    """Serialise the fitted pipeline to ``path`` (joblib, pickle protocol 5)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path, protocol=5)


def save_metrics(metrics: dict[str, float], path: Path = METRICS_PATH) -> None:
    """Write the evaluation metrics to ``path`` as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def run_training_pipeline(
    feature_path: Path = FEATURE_PARQUET_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    *,
    param_grid: dict[str, list[Any]] | None = None,
    validate_split: bool = True,
) -> tuple[TransformedTargetRegressor, dict[str, float]]:
    """Load features, train + tune the model, evaluate it and persist artifacts.

    Args:
        feature_path: Source feature Parquet file.
        model_path: Destination for the serialised pipeline.
        metrics_path: Destination for the evaluation-metrics JSON.
        param_grid: Optional override for the hyper-parameter grid (see
            :func:`train_model`).
        validate_split: When ``True`` (default) vet the train/test split for
            leakage and distribution mismatch before training (issue #25);
            a failing check raises
            ``split_validation.TrainTestSplitValidationError``.

    Returns:
        ``(fitted_model, metrics)``.
    """
    logger.info("Loading feature table from %s", feature_path)
    feature_table = load_feature_table(feature_path)
    logger.info("Feature table: %d rows x %d columns", *feature_table.shape)

    features, target = split_features_target(feature_table)
    x_train, x_test, y_train, y_test = split_train_test(features, target)
    logger.info("Train rows: %d | test rows: %d", len(x_train), len(x_test))

    if validate_split:
        validate_train_test_split((x_train, y_train), (x_test, y_test))
        logger.info("Train/test split passed leakage and distribution checks")

    logger.info("Training model (grid search over %s)", param_grid or DEFAULT_PARAM_GRID)
    model = train_model(x_train, y_train, param_grid=param_grid)

    metrics = evaluate_model(model, x_test, y_test)
    logger.info("Test metrics: %s", metrics)

    save_model(model, model_path)
    logger.info("Wrote fitted model to %s", model_path)
    save_metrics(metrics, metrics_path)
    logger.info("Wrote evaluation metrics to %s", metrics_path)
    return model, metrics


def main() -> None:
    """Console entry point: configure logging and run the pipeline once."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    run_training_pipeline()


if __name__ == "__main__":
    main()
