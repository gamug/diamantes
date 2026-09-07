"""Diamond feature-engineering library.

Pure, side-effect-free transformations that turn the raw course dataset
(``data/01_raw/diamantes.csv``) into a model-ready feature table. The
orchestration around these functions (file IO, logging, CLI) lives in
:mod:`feature_pipeline`.

The rules encoded here come from the exploratory notebooks
``notebooks/3-analysis/02-jrz-data_description_Manual-pandas-2024_10_24.ipynb``
and ``notebooks/4-feat_eng/01-gmg-basic-feature-engineering-pipeline-2026_08_18.ipynb``
and from the data dictionary ``data/01_raw/datos_diamantes_Info.txt``.

Train/test split, scaling and encoding are intentionally **not** done here:
they require statistics fitted on the training split only and belong to the
training pipeline, so keeping them out of the feature table prevents
test-set information from leaking into stored features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Column groups -----------------------------------------------------
#: Numeric columns stored as text (and seeded with garbage) in the raw file.
NUMERIC_COLUMNS: list[str] = ["carat", "depth", "table", "price", "x", "y", "z"]

#: Physical length columns (mm) collapsed into ``volume``.
DIMENSION_COLUMNS: list[str] = ["x", "y", "z"]

#: Regression target / label carried into the feature table.
PRICE_COLUMN: str = "price"

#: Engineered feature added by :func:`add_volume_feature`.
VOLUME_COLUMN: str = "volume"

#: Feature-table columns, fixed order: raw inputs + ``volume`` + label.
FEATURE_TABLE_COLUMNS: list[str] = [
    "carat",
    "cut",
    "color",
    "clarity",
    "depth",
    "table",
    "x",
    "y",
    "z",
    VOLUME_COLUMN,
    PRICE_COLUMN,
]

# --- Domain rules (data/01_raw/datos_diamantes_Info.txt) -------------
#: Known category grades, worst -> best. Anything else is corrupted.
KNOWN_CATEGORIES: dict[str, list[str]] = {
    "cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
    "color": ["J", "I", "H", "G", "F", "E", "D"],
    "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
}

#: Inclusive ``(min, max)`` physical range documented for each measurement.
#: Values outside these are data-entry errors, not extreme-but-real stones.
MEASUREMENT_BOUNDS: dict[str, tuple[float, float]] = {
    "carat": (0.2, 5.01),
    "depth": (43.0, 79.0),
    "table": (43.0, 95.0),
    "x": (0.0, 10.74),
    "y": (0.0, 58.9),
    "z": (0.0, 31.8),
}

#: Max allowed disagreement (percentage points) between the recorded ``depth``
#: and the depth percentage implied by the geometry ``2*z / (x + y) * 100`` --
#: ``depth``'s own definition in the data dictionary. Rows beyond this are
#: internally inconsistent records (a mistyped dimension) rather than real
#: stones and are dropped by :func:`drop_geometry_inconsistent_rows`.
GEOMETRY_DEPTH_TOLERANCE_PP: float = 1.0

#: Final dtypes for the stored feature table.
FEATURE_TABLE_DTYPES: dict[str, str] = {
    "carat": "float32",
    "cut": "category",
    "color": "category",
    "clarity": "category",
    "depth": "float32",
    "table": "float32",
    "x": "float32",
    "y": "float32",
    "z": "float32",
    VOLUME_COLUMN: "float32",
    PRICE_COLUMN: "int64",
}


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the text numeric columns to numbers; unparseable values -> ``NaN``.

    Args:
        df: Raw table (all columns object/text). Not mutated.

    Returns:
        A copy where every column in :data:`NUMERIC_COLUMNS` is numeric and any
        value that could not be parsed (``"wetf"``, ``"23567364"``, ``""``) is
        ``NaN``.
    """
    df = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def restrict_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace category values outside their known grade vocabulary with ``NaN``.

    Args:
        df: Table containing ``cut``/``color``/``clarity``. Not mutated.

    Returns:
        A copy where any ``cut``/``color``/``clarity`` value absent from
        :data:`KNOWN_CATEGORIES` is ``NaN``.
    """
    df = df.copy()
    for column, known_values in KNOWN_CATEGORIES.items():
        if column in df.columns:
            df.loc[~df[column].isin(known_values), column] = np.nan
    return df


def apply_domain_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Null out measurements outside their documented physical range.

    Args:
        df: Table with numeric measurement columns (run
            :func:`coerce_numeric_columns` first). Not mutated.

    Returns:
        A copy where values outside :data:`MEASUREMENT_BOUNDS` are ``NaN`` and
        zero/negative ``x``/``y``/``z`` lengths (physically impossible) are
        ``NaN`` too.
    """
    df = df.copy()
    for column, (low, high) in MEASUREMENT_BOUNDS.items():
        if column in df.columns:
            df.loc[~df[column].between(low, high), column] = np.nan
    for column in DIMENSION_COLUMNS:
        if column in df.columns:
            df.loc[df[column] <= 0, column] = np.nan
    return df


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop incomplete, duplicated and column-swap-corrupted rows.

    - Any row with a missing value is dropped: after cleaning, a gap means the
      value was corrupted and can't be recovered.
    - Exact duplicate rows are dropped so the same stone can't leak across a
      later train/test split.
    - Rows where ``z == carat`` or ``y == depth`` are dropped: these equalities
      are physically meaningless and flag two columns swapped during data
      entry (found in the manual EDA notebook).

    Args:
        df: Cleaned table. Not mutated.

    Returns:
        A copy with the offending rows removed and the index reset.
    """
    cleaned = df.dropna().drop_duplicates()
    swapped = (cleaned["z"] == cleaned["carat"]) | (cleaned["y"] == cleaned["depth"])
    return cleaned.loc[~swapped].reset_index(drop=True)


def implied_depth_percentage(df: pd.DataFrame) -> pd.Series:
    """Depth percentage implied by a stone's geometry: ``2*z / (x + y) * 100``.

    This is the definition of ``depth`` given in the data dictionary
    (``data/01_raw/datos_diamantes_Info.txt``).

    Args:
        df: Table with numeric ``x``, ``y`` and ``z`` columns.

    Returns:
        The implied depth percentage, one value per row.
    """
    return 2.0 * df["z"] / (df["x"] + df["y"]) * 100.0


def drop_geometry_inconsistent_rows(
    df: pd.DataFrame, tolerance_pp: float = GEOMETRY_DEPTH_TOLERANCE_PP
) -> pd.DataFrame:
    """Drop rows whose recorded ``depth`` contradicts their ``x``/``y``/``z`` geometry.

    ``depth`` is *defined* as ``2*z / (x + y) * 100``. A recorded value that
    disagrees with that formula by more than ``tolerance_pp`` percentage points
    is an internally inconsistent record (e.g. a mistyped dimension), not a
    real stone, so it is removed before the feature table is built. Every
    individual value in such a row can still sit inside its own documented
    range, which is why :func:`apply_domain_bounds` does not catch it.

    Args:
        df: Table with numeric ``depth``/``x``/``y``/``z`` and no missing
            values in those columns. Not mutated.
        tolerance_pp: Allowed absolute disagreement, in percentage points.

    Returns:
        A copy without the inconsistent rows, index reset.
    """
    disagreement = (df["depth"] - implied_depth_percentage(df)).abs()
    return df.loc[disagreement <= tolerance_pp].reset_index(drop=True)


def add_volume_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``volume = x * y * z``.

    The manual EDA found ``carat``, ``x``, ``y`` and ``z`` to be near-duplicate
    size measures (pairwise correlation up to 0.98). One ``volume`` feature
    keeps almost all of that information next to ``carat`` instead of three
    collinear columns.

    Args:
        df: Table containing ``x``, ``y`` and ``z``. Not mutated.

    Returns:
        A copy with a ``volume`` column appended.
    """
    df = df.copy()
    df[VOLUME_COLUMN] = df["x"] * df["y"] * df["z"]
    return df


def cast_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast the feature table to its final dtypes.

    Args:
        df: Fully cleaned table with the :data:`FEATURE_TABLE_COLUMNS` columns
            and no missing values. Not mutated.

    Returns:
        A copy cast to :data:`FEATURE_TABLE_DTYPES`.
    """
    return df.astype(
        {col: dtype for col, dtype in FEATURE_TABLE_DTYPES.items() if col in df.columns}
    )


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Run the full raw -> feature-table transformation in memory.

    Args:
        raw_df: Raw table as read from ``data/01_raw/diamantes.csv`` (columns
            as text).

    Returns:
        The model-ready feature table: columns :data:`FEATURE_TABLE_COLUMNS`,
        no missing values, no duplicates, no geometry-inconsistent rows, dtypes
        per :data:`FEATURE_TABLE_DTYPES`.
    """
    df = coerce_numeric_columns(raw_df)
    df = restrict_categorical_columns(df)
    df = apply_domain_bounds(df)
    df = drop_invalid_rows(df)
    df = drop_geometry_inconsistent_rows(df)
    df = add_volume_feature(df)
    df = df[FEATURE_TABLE_COLUMNS]
    return cast_feature_dtypes(df)


#: Engineered feature columns without the ``price`` label -- the shape the
#: inference pipeline feeds to a trained model.
INFERENCE_FEATURE_COLUMNS: list[str] = [
    column for column in FEATURE_TABLE_COLUMNS if column != PRICE_COLUMN
]


def build_inference_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw new diamonds into model inputs, keeping every input row.

    Same cleaning and engineering as :func:`build_features` (type-fix text
    numeric columns, null out values outside the documented ranges or category
    vocabularies, engineer ``volume = x * y * z``) **except** the row-dropping
    steps: at inference time we want one prediction per input row, so invalid /
    missing values are left as ``NaN`` for the model pipeline's imputers to
    handle rather than removed. The ``price`` label is not required.

    Args:
        raw_df: New data with the raw column layout (``carat``, ``cut``,
            ``color``, ``clarity``, ``depth``, ``table``, ``x``, ``y``, ``z``;
            ``price`` optional), values as text like the source CSV.

    Returns:
        A copy with the :data:`INFERENCE_FEATURE_COLUMNS` columns, one row per
        input row, cast to :data:`FEATURE_TABLE_DTYPES`.
    """
    df = coerce_numeric_columns(raw_df)
    df = restrict_categorical_columns(df)
    df = apply_domain_bounds(df)
    df = add_volume_feature(df)
    df = df[INFERENCE_FEATURE_COLUMNS]
    return cast_feature_dtypes(df)
