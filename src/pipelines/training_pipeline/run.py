"""Training pipeline: engineered features -> a trained, evaluated, persisted model.

Orchestrates :mod:`model.train`, :mod:`model.evaluate` and
:mod:`model.persistence` to reproduce, as a single runnable script,
``notebooks/5-models/train_model_pipeline.py`` (confirmed as the best model
family/hyperparameters by
``notebooks/5-models/02-gmg-basic_algorithms_model_selection-...`` and
``03-gmg-first_model-...``).

Run with (``src`` must be importable, matching this project's pytest
``pythonpath`` convention — see ``CLAUDE.md``)::

    PYTHONPATH=src uv run python -m pipelines.training_pipeline.run

TODO(tomorrow): once :mod:`model.validation` is implemented, run
``check_model_evaluation`` here and gate the ``save_model`` call on it
passing.
"""

from pathlib import Path
from typing import Any

from data.constants import FEATURE_COLUMNS, TARGET_COLUMN
from data.extraction import load_parquet
from model.evaluate import regression_metrics
from model.persistence import save_model
from model.train import split_train_test, train_model

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

FEATURE_PARQUET_PATH = DATA_DIR / "04_feature" / "diamantes_clean.parquet"
MODEL_DIR = DATA_DIR / "06_models"
MODEL_FILE_NAME = "diamantes_price-hist_gradient_boosting-v1.joblib"


def run(
    feature_parquet_path: Path = FEATURE_PARQUET_PATH,
    model_dir: Path = MODEL_DIR,
    model_file_name: str = MODEL_FILE_NAME,
    hyperparameters: dict[str, list[Any]] | None = None,
    cv: int = 5,
) -> dict[str, float]:
    """Run the full training pipeline: split, tune, evaluate and persist the model.

    Args:
        feature_parquet_path: Engineered features to train on
            (``data/04_feature/diamantes_clean.parquet``).
        model_dir: Directory to save the trained model into
            (``data/06_models``).
        model_file_name: Filename for the saved model artifact.
        hyperparameters: Grid search space, forwarded to
            :func:`model.train.train_model`. Defaults to
            :data:`model.train.DEFAULT_HYPERPARAMETER_GRID`; override with a
            smaller grid for fast/test runs.
        cv: Number of cross-validation folds, forwarded to
            :func:`model.train.train_model`.

    Returns:
        The held-out test-set regression metrics (see
        :func:`model.evaluate.regression_metrics`).
    """
    dataset = load_parquet(feature_parquet_path)
    x = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = split_train_test(x, y)
    print(f"x_train: {x_train.shape}  x_test: {x_test.shape}")

    grid_search = train_model(x_train, y_train, hyperparameters=hyperparameters, cv=cv)
    best_model = grid_search.best_estimator_
    print(f"Best hyperparameters: {grid_search.best_params_}")

    y_pred = best_model.predict(x_test)
    metrics = regression_metrics(y_test, y_pred)
    print(f"Test metrics: {metrics}")

    save_model(best_model, model_dir / model_file_name)
    print(f"Saved model to {model_dir / model_file_name}")

    return metrics


if __name__ == "__main__":
    run()
