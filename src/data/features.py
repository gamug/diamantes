"""Feature engineering: row-level cleanup and the model preprocessing pipeline.

Ported from ``notebooks/4-feat_eng/01-gmg-basic-feature-engineering-pipeline-2026_08_18.ipynb``.

``compute_volume`` and ``volume_feature_names_out`` are defined at module
level (not as closures/lambdas) so that a fitted preprocessor containing them
inside a ``FunctionTransformer`` can be pickled and unpickled with joblib:
joblib/pickle references a function by its module + qualified name, so the
same names must resolve to the same functions wherever the model is loaded
(training, inference, the Streamlit apps). See ``src/README.md`` and the
project's pre-commit/joblib notes in memory for this exact gotcha.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler

from data.constants import (
    CATEGORICAL_ORDINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_CATEGORIES,
    VOLUME_SOURCE_FEATURES,
)


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully-duplicated rows.

    Args:
        df: Input DataFrame.

    Returns:
        ``df`` without duplicate rows, with the index reset.
    """
    return df.drop_duplicates().reset_index(drop=True)


def remove_corrupted_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a ``z``/``carat`` or ``y``/``depth`` column-swap data-entry error.

    The EDA (``notebooks/2-exploration``) found a handful of rows where
    ``z == carat`` or ``y == depth`` exactly — a strong signal of two columns
    having been swapped during data entry rather than a genuine measurement.

    Args:
        df: Input DataFrame; must contain ``carat``, ``depth``, ``y`` and
            ``z`` columns.

    Returns:
        ``df`` without the corrupted rows, with the index reset.
    """
    mask_corrupt = (df["z"] == df["carat"]) | (df["y"] == df["depth"])
    return df.loc[~mask_corrupt].reset_index(drop=True)


def compute_volume(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Collapse the ``x``, ``y``, ``z`` dimension columns into a single ``volume`` column.

    Args:
        x: A ``(n_samples, 3)`` array with columns ``x``, ``y``, ``z`` in
            that order.

    Returns:
        A ``(n_samples, 1)`` array with ``volume = x * y * z``.
    """
    length, width, depth = x[:, 0], x[:, 1], x[:, 2]
    return (length * width * depth).reshape(-1, 1)


def volume_feature_names_out(
    transformer: FunctionTransformer, input_features: list[str]
) -> npt.NDArray[np.str_]:
    """Named (picklable) replacement for a lambda: always outputs one column, ``"volume"``.

    Args:
        transformer: The ``FunctionTransformer`` calling this (unused; the
            signature is required by scikit-learn's ``feature_names_out``
            callable convention).
        input_features: Input feature names (unused, same reason).

    Returns:
        ``array(["volume"])``.
    """
    return np.array(["volume"])


def build_preprocessor(
    numeric_features: list[str] | None = None,
    volume_source_features: list[str] | None = None,
    categorical_ordinal_features: list[str] | None = None,
    ordinal_categories: dict[str, list[str]] | None = None,
) -> ColumnTransformer:
    """Build the diamonds feature-preprocessing ``ColumnTransformer``.

    - Numeric features: median-imputed, then standard-scaled.
    - Volume source features (``x``, ``y``, ``z``): median-imputed, collapsed
      into a single ``volume = x * y * z`` feature, then standard-scaled —
      this replaces the raw, near-collinear dimensions (correlated with
      ``carat`` up to r=0.98 per the EDA).
    - Categorical ordinal features (``cut``, ``color``, ``clarity``):
      most-frequent-imputed, then ordinal-encoded worst -> best so a higher
      code always means a higher expected price premium.

    Args:
        numeric_features: Defaults to
            :data:`src.data.constants.NUMERIC_FEATURES`.
        volume_source_features: Defaults to
            :data:`src.data.constants.VOLUME_SOURCE_FEATURES`.
        categorical_ordinal_features: Defaults to
            :data:`src.data.constants.CATEGORICAL_ORDINAL_FEATURES`.
        ordinal_categories: Defaults to
            :data:`src.data.constants.ORDINAL_CATEGORIES`.

    Returns:
        An unfitted ``ColumnTransformer`` ready to be used inside a model
        pipeline (see :mod:`src.model.train`).
    """
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    volume_source_features = (
        VOLUME_SOURCE_FEATURES if volume_source_features is None else volume_source_features
    )
    categorical_ordinal_features = (
        CATEGORICAL_ORDINAL_FEATURES
        if categorical_ordinal_features is None
        else categorical_ordinal_features
    )
    ordinal_categories = ORDINAL_CATEGORIES if ordinal_categories is None else ordinal_categories

    volume_transformer = FunctionTransformer(
        compute_volume,
        feature_names_out=volume_feature_names_out,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    volume_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("volume", volume_transformer),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[ordinal_categories[col] for col in categorical_ordinal_features]
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("volume", volume_pipeline, volume_source_features),
            ("cat_ordinal", categorical_ordinal_pipeline, categorical_ordinal_features),
        ]
    )
