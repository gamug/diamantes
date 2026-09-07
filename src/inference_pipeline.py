"""Inference pipeline: trained model + new diamonds -> predicted prices.

Standalone, autonomously runnable script for issue #27 ("6. Inference pipeline
development"), part of the FTI scripts architecture (issue #15).

It:

1. **loads** the fitted pipeline saved by the training pipeline (issue #24)
   from ``data/06_models/`` (a ``TransformedTargetRegressor`` that already
   carries the impute / scale / ordinal-encode / log-price steps);
2. **reads** new data from a CSV with the raw column layout (same as
   ``data/01_raw/diamantes.csv``);
3. **applies the same transformations used in training** -- the cleaning and
   ``volume`` engineering from :func:`diamond_features.build_inference_features`
   (row-preserving, so every input row gets a prediction) followed by the
   model's own preprocessing;
4. **stores and visualizes** the predictions:
   ``data/07_model_output/diamantes_predictions.parquet`` (input columns +
   ``predicted_price``) and ``diamantes_predictions.png``.

Run it (``src`` is on ``sys.path`` via ``pyproject.toml``'s
``[tool.pytest.ini_options] pythonpath``; for a bare ``python`` call, either
``cd src`` or set ``PYTHONPATH=src``)::

    uv run python src/inference_pipeline.py
    uv run python -m inference_pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from matplotlib.figure import Figure  # object-oriented API: no global backend, no pyplot state
from sklearn.base import BaseEstimator

from diamond_features import build_inference_features
from feature_pipeline import load_raw_data
from training_pipeline import MODEL_FEATURES, MODEL_PATH

logger = logging.getLogger(__name__)

#: Repository root: this file lives at ``src/inference_pipeline.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
#: Default "new data" to score. The raw course CSV stands in for a real new
#: batch; any CSV with the same raw column layout works.
INPUT_CSV_PATH = DATA_DIR / "01_raw" / "diamantes.csv"
#: Prediction results (Kedro layer 07_model_output).
PREDICTIONS_PATH = DATA_DIR / "07_model_output" / "diamantes_predictions.parquet"
PREDICTIONS_PLOT_PATH = DATA_DIR / "07_model_output" / "diamantes_predictions.png"

#: Column appended to the input rows with the model's price estimate.
PREDICTION_COLUMN = "predicted_price"
#: Raw columns the transformation needs (``price`` is not required for inference).
REQUIRED_INPUT_COLUMNS: list[str] = [
    "carat",
    "cut",
    "color",
    "clarity",
    "depth",
    "table",
    "x",
    "y",
    "z",
]


def load_model(path: Path = MODEL_PATH) -> BaseEstimator:
    """Load the fitted model pipeline saved by the training pipeline.

    Args:
        path: Path to the ``.joblib`` artifact in ``data/06_models/``.

    Returns:
        The deserialised estimator (a fitted ``TransformedTargetRegressor``).

    Raises:
        FileNotFoundError: If ``path`` does not exist -- run the training
            pipeline (issue #24) first.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {path}. Run the training pipeline (issue #24) first."
        )
    model = load(path)
    logger.info("Loaded %s from %s", type(model).__name__, path)
    return model


def load_new_data(csv_path: Path = INPUT_CSV_PATH) -> pd.DataFrame:
    """Read the new-data CSV with no type coercion (columns stay text).

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    return load_raw_data(csv_path)


def prepare_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw new data into the model-input matrix (one row per input row).

    Args:
        raw_df: New data with the raw column layout, values as text.

    Returns:
        A frame with exactly the :data:`training_pipeline.MODEL_FEATURES`
        columns, same row count as ``raw_df``.

    Raises:
        ValueError: If a required raw column is missing.
    """
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in raw_df.columns]
    if missing:
        raise ValueError(f"new data is missing required columns: {missing}")
    return build_inference_features(raw_df)[MODEL_FEATURES]


def predict_prices(model: BaseEstimator, features: pd.DataFrame) -> np.ndarray:
    """Predict prices for a prepared feature matrix."""
    return np.asarray(model.predict(features), dtype=float)


def build_predictions_frame(raw_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Attach the predictions to the original input rows."""
    result = raw_df.reset_index(drop=True).copy()
    result[PREDICTION_COLUMN] = predictions
    return result


def save_predictions(df: pd.DataFrame, path: Path = PREDICTIONS_PATH) -> None:
    """Write the predictions table to Parquet, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def plot_predictions(df: pd.DataFrame, path: Path = PREDICTIONS_PLOT_PATH) -> None:
    """Save a two-panel PNG: predicted-price histogram and price-vs-carat scatter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(9.0, 4.0), layout="constrained")
    hist_ax, scatter_ax = fig.subplots(1, 2)

    prices = df[PREDICTION_COLUMN]
    hist_ax.hist(prices.dropna(), bins=40, color="#4c72b0")
    hist_ax.set_xlabel("predicted price (USD)")
    hist_ax.set_ylabel("diamonds")
    hist_ax.set_title("Predicted price distribution")

    scatter_ax.scatter(
        pd.to_numeric(df["carat"], errors="coerce"), prices, s=6, alpha=0.3, color="#55a868"
    )
    scatter_ax.set_xlabel("carat")
    scatter_ax.set_ylabel("predicted price (USD)")
    scatter_ax.set_title("Predicted price vs. carat")

    fig.suptitle("Diamond price predictions")
    fig.savefig(path, dpi=120)


def run_inference_pipeline(
    input_csv: Path = INPUT_CSV_PATH,
    model_path: Path = MODEL_PATH,
    output_path: Path = PREDICTIONS_PATH,
    plot_path: Path = PREDICTIONS_PLOT_PATH,
) -> pd.DataFrame:
    """Load the model, score ``input_csv`` and persist + plot the predictions.

    Args:
        input_csv: New data CSV (raw column layout).
        model_path: Fitted model artifact.
        output_path: Destination Parquet file for input rows + ``predicted_price``.
        plot_path: Destination PNG for the prediction plots.

    Returns:
        The predictions table (also written to ``output_path``).
    """
    model = load_model(model_path)
    raw_df = load_new_data(input_csv)
    logger.info("New data: %d rows x %d columns", *raw_df.shape)

    features = prepare_features(raw_df)
    predictions = predict_prices(model, features)
    logger.info(
        "Predicted price: min %.0f | mean %.0f | max %.0f",
        np.nanmin(predictions),
        np.nanmean(predictions),
        np.nanmax(predictions),
    )

    result = build_predictions_frame(raw_df, predictions)
    save_predictions(result, output_path)
    logger.info("Wrote %d predictions to %s", len(result), output_path)
    plot_predictions(result, plot_path)
    logger.info("Wrote predictions plot to %s", plot_path)
    return result


def main() -> None:
    """Console entry point: configure logging and run the pipeline once."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    run_inference_pipeline()


if __name__ == "__main__":
    main()
