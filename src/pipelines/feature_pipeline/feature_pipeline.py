"""Feature pipeline: raw diamonds CSV -> model-ready feature table.

Standalone, autonomously runnable script for issue #22 ("1. Feature pipeline
development"), part of the FTI scripts architecture (issue #15). It reproduces,
as plain Python, the data-preparation and feature-engineering steps explored in
``notebooks/3-analysis/02-jrz-data_description_Manual-pandas-2024_10_24.ipynb``
and ``notebooks/4-feat_eng/01-gmg-basic-feature-engineering-pipeline-2026_08_18.ipynb``.

Pipeline stages (see :func:`build_features`):

1. **Load** the raw CSV exactly as supplied for the course
   (``data/01_raw/diamantes.csv`` -- never the original Kaggle dataset).
2. **Type-fix** the numeric columns: the course file stores them as text and
   seeds them with non-numeric garbage (e.g. ``"23567364"`` in ``cut``,
   ``"wetf"`` in ``color``), so every non-parseable value becomes ``NaN``.
3. **Validate** against the data dictionary
   (``data/01_raw/datos_diamantes_Info.txt``): categorical grades outside their
   known vocabulary and measurements outside their documented physical range
   become ``NaN``; zero/negative ``x``/``y``/``z`` lengths are impossible and
   are nulled too.
4. **Drop** rows that are still incomplete, exact duplicates, or show a
   ``z == carat`` / ``y == depth`` column-swap data-entry error.
5. **Engineer** ``volume = x * y * z`` (a single size feature to sit alongside
   ``carat`` instead of the three near-collinear raw dimensions).
6. **Cast** to final dtypes and **write** the feature table to
   ``data/04_feature/diamantes_features.parquet``.

No train/test split, scaling or encoding happens here: those need statistics
fitted on the training split only and belong to the training pipeline, so
keeping them out avoids leaking test data into the stored features.

Run it (``src`` is on ``PYTHONPATH`` via ``pyproject.toml``'s pytest config;
for a bare ``python`` call, either ``cd src`` or set ``PYTHONPATH=src``)::

    uv run python src/pipelines/feature_pipeline/feature_pipeline.py
    uv run python -m pipelines.feature_pipeline.feature_pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Paths ------------------------------------------------------------------
#: Repository root: ``src/pipelines/feature_pipeline/feature_pipeline.py`` -> up 3.
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RAW_CSV_PATH = DATA_DIR / "01_raw" / "diamantes.csv"
FEATURE_PARQUET_PATH = DATA_DIR / "04_feature" / "diamantes_features.parquet"

# --- Column groups --------------------------------------------------------
#: Numeric columns stored as text in the raw file and coerced back to numbers.
NUMERIC_COLUMNS: list[str] = ["carat", "depth", "table", "price", "x", "y", "z"]

#: Physical length columns (mm) that are collapsed into ``volume``.
DIMENSION_COLUMNS: list[str] = ["x", "y", "z"]

#: Regression target / label carried into the feature table.
PRICE_COLUMN: str = "price"

#: Engineered feature added by :func:`add_volume_feature`.
VOLUME_COLUMN: str = "volume"

#: Columns of the feature table, in a fixed order (raw inputs + ``volume`` +
#: label). Every downstream stage selects the subset it needs from this.
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

# --- Domain rules (data/01_raw/datos_diamantes_Info.txt) ----------------
#: Known category grades, worst -> best. Anything else is treated as corrupted.
KNOWN_CATEGORIES: dict[str, list[str]] = {
    "cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
    "color": ["J", "I", "H", "G", "F", "E", "D"],
    "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
}

#: Inclusive ``(min, max)`` physical range documented for each measurement.
#: Values outside these bounds are data-entry errors, not extreme-but-real
#: stones, so they are nulled rather than kept.
MEASUREMENT_BOUNDS: dict[str, tuple[float, float]] = {
    "carat": (0.2, 5.01),
    "depth": (43.0, 79.0),
    "table": (43.0, 95.0),
    "x": (0.0, 10.74),
    "y": (0.0, 58.9),
    "z": (0.0, 31.8),
}

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


def load_raw_data(csv_path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Load the raw diamonds CSV with no type coercion.

    Args:
        csv_path: Path to ``data/01_raw/diamantes.csv`` (the modified file
            supplied for the course, *not* the original Kaggle dataset).

    Returns:
        The raw table, every column read as-is so the type-fixing step can
        inspect the original string values.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Raw data file not found: {csv_path}")
    return pd.read_csv(csv_path, dtype="object", low_memory=False)


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the text numeric columns to numbers, garbage values becoming ``NaN``.

    Args:
        df: Raw table from :func:`load_raw_data`. Not mutated.

    Returns:
        A copy where every column in :data:`NUMERIC_COLUMNS` is numeric and any
        value that could not be parsed (e.g. ``"wetf"``, ``"23567364"``) is
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
        A copy where any ``cut``/``color``/``clarity`` value not present in
        :data:`KNOWN_CATEGORIES` has been set to ``NaN``.
    """
    df = df.copy()
    for column, known_values in KNOWN_CATEGORIES.items():
        if column in df.columns:
            df.loc[~df[column].isin(known_values), column] = np.nan
    return df


def apply_domain_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Null out measurements that fall outside their documented physical range.

    Args:
        df: Table with numeric measurement columns (run
            :func:`coerce_numeric_columns` first). Not mutated.

    Returns:
        A copy where values outside :data:`MEASUREMENT_BOUNDS` are ``NaN``, and
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

    - Rows with any missing value are dropped: after the cleaning steps a gap
      means the value was corrupted and cannot be recovered.
    - Exact duplicate rows are dropped so the same stone cannot leak across a
      later train/test split.
    - Rows where ``z == carat`` or ``y == depth`` are dropped: these equalities
      have no physical meaning and flag two columns swapped during data entry
      (identified in the manual EDA notebook).

    Args:
        df: Cleaned table. Not mutated.

    Returns:
        A copy with the offending rows removed and the index reset.
    """
    cleaned = df.dropna().drop_duplicates()
    swapped = (cleaned["z"] == cleaned["carat"]) | (cleaned["y"] == cleaned["depth"])
    return cleaned.loc[~swapped].reset_index(drop=True)


def add_volume_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``volume = x * y * z``.

    The manual EDA found ``carat``, ``x``, ``y`` and ``z`` to be near-duplicate
    size measures (pairwise correlation up to 0.98). A single ``volume``
    feature keeps almost all of that information for a downstream model to use
    next to ``carat``, instead of three collinear columns.

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
        raw_df: Raw table as returned by :func:`load_raw_data`.

    Returns:
        The model-ready feature table: columns :data:`FEATURE_TABLE_COLUMNS`,
        no missing values, no duplicates, dtypes per
        :data:`FEATURE_TABLE_DTYPES`.
    """
    df = coerce_numeric_columns(raw_df)
    df = restrict_categorical_columns(df)
    df = apply_domain_bounds(df)
    df = drop_invalid_rows(df)
    df = add_volume_feature(df)
    df = df[FEATURE_TABLE_COLUMNS]
    return cast_feature_dtypes(df)


def save_features(df: pd.DataFrame, path: Path = FEATURE_PARQUET_PATH) -> None:
    """Write the feature table to Parquet, creating parent directories.

    Args:
        df: Feature table from :func:`build_features`.
        path: Destination ``.parquet`` file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run_feature_pipeline(
    raw_csv_path: Path = RAW_CSV_PATH,
    output_path: Path = FEATURE_PARQUET_PATH,
) -> pd.DataFrame:
    """Load the raw CSV, build the feature table and persist it.

    Args:
        raw_csv_path: Source raw CSV.
        output_path: Destination Parquet file for the feature table.

    Returns:
        The feature table that was written (also handy for tests / notebooks).
    """
    logger.info("Loading raw data from %s", raw_csv_path)
    raw_df = load_raw_data(raw_csv_path)
    logger.info("Raw data: %d rows x %d columns", raw_df.shape[0], raw_df.shape[1])

    features = build_features(raw_df)
    dropped = raw_df.shape[0] - features.shape[0]
    logger.info(
        "Feature table: %d rows x %d columns (%d raw rows dropped as invalid/duplicate)",
        features.shape[0],
        features.shape[1],
        dropped,
    )

    save_features(features, output_path)
    logger.info("Wrote feature table to %s", output_path)
    return features


def main() -> None:
    """Console entry point: configure logging and run the pipeline once."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    run_feature_pipeline()


if __name__ == "__main__":
    main()
