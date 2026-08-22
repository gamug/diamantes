"""Prediction serving: load a trained model and score new diamonds.

Business logic behind the Streamlit apps in ``notebooks/7-deploy`` — kept
independent of any UI framework so it can be unit-tested and reused from
either the interactive or batch app, a CLI, or a future API.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from data.constants import (
    CATEGORICAL_ORDINAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    ORDINAL_CATEGORIES,
    VOLUME_SOURCE_FEATURES,
)
from model.persistence import load_model

#: Filename of the production model artifact, relative to ``data/06_models``.
MODEL_FILE_NAME = "diamantes_price-hist_gradient_boosting-v1.joblib"

#: Diamonds below this carat weight fall outside the training data's range
#: (see ``notebooks/2-exploration``); predictions for them are extrapolations.
MIN_TRAINED_CARAT = 1.0


def load_production_model(model_dir: Path) -> Any:
    """Load the production diamond-price model.

    Args:
        model_dir: Directory containing the model artifact, conventionally
            ``data/06_models``.

    Returns:
        The fitted model/pipeline (a ``TransformedTargetRegressor``).
    """
    return load_model(model_dir / MODEL_FILE_NAME)


def predict_price(model: Any, diamonds: pd.DataFrame) -> npt.NDArray[np.floating]:
    """Predict the price for one or more diamonds.

    Args:
        model: A fitted model/pipeline exposing ``.predict``.
        diamonds: DataFrame with (at least) the columns in
            :data:`data.constants.FEATURE_COLUMNS`.

    Returns:
        Predicted price per row.
    """
    predictions: npt.NDArray[np.floating] = model.predict(diamonds[FEATURE_COLUMNS])
    return predictions


def is_out_of_training_range(diamonds: pd.DataFrame) -> pd.Series:
    """Flag rows whose ``carat`` falls outside the training data's range.

    Args:
        diamonds: DataFrame with a ``carat`` column.

    Returns:
        Boolean series, ``True`` where the prediction is an extrapolation.
    """
    return diamonds["carat"] < MIN_TRAINED_CARAT


def preprocess_batch_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a user-uploaded batch CSV's columns to what the model expects.

    Column names are lower-cased/stripped, ordinal grade values (``cut``,
    ``color``, ``clarity``) are matched case-insensitively against the known
    categories, and numeric columns are coerced (invalid values become
    ``NaN`` rather than raising).

    Args:
        df: The DataFrame read from the uploaded CSV, as-is.

    Returns:
        A normalized copy of ``df``, ready for :func:`predict_price` (once
        any missing required columns are added).
    """
    processed_df = df.copy()
    processed_df.columns = [str(c).strip().lower() for c in processed_df.columns]

    for grade_col in CATEGORICAL_ORDINAL_FEATURES:
        if grade_col not in processed_df.columns:
            continue
        lookup = {c.lower(): c for c in ORDINAL_CATEGORIES[grade_col]}
        processed_df[grade_col] = (
            processed_df[grade_col]
            .astype(str)
            .str.strip()
            .map(lambda v, lookup=lookup: lookup.get(v.lower(), v))
        )

    numeric_columns = NUMERIC_FEATURES + VOLUME_SOURCE_FEATURES
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

    return processed_df
